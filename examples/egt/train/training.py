
from mpi4py import MPI
import numpy as np
from collections import OrderedDict
import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader, Sampler
from contextlib import nullcontext

import yaml
from yaml import SafeLoader as yaml_Loader, SafeDumper as yaml_Dumper
import os,sys

from tqdm import tqdm

from gammagl.utils.dotdict import HDict
HDict.L.update_globals({'path':os.path})

def str_presenter(dumper, data):
  if len(data.splitlines()) > 1:  # check for multiline string
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
  return dumper.represent_scalar('tag:yaml.org,2002:str', data)
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)


def read_config_from_file(config_file):
    with open(config_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml_Loader)

def save_config_to_file(config, config_file):
    with open(config_file, 'w') as fp:
        return yaml.dump(config, fp, sort_keys=False, Dumper=yaml_Dumper)


class StopTrainingException(Exception):
    pass

class CollatedBatch(list):
    pass

class DistributedTestDataSampler(Sampler):
    def __init__(self, data_source, batch_size, rank, world_size):
        data_len = len(data_source)
        all_indices = np.arange(data_len, dtype=int)
        split_indices = np.array_split(all_indices, world_size)
        
        num_batches = (len(split_indices[0]) + batch_size -1) // batch_size
        self.batch_indices = [i.tolist() for i in np.array_split(split_indices[rank],
                                                                 num_batches)]
    
    def __iter__(self):
        return iter(self.batch_indices)
    
    def __len__(self):
        return len(self.batch_indices)



def cached_property(func):
    atrribute_name = f'_{func.__name__}'
    def _wrapper(self):
        try:
            return getattr(self, atrribute_name)
        except AttributeError:
            val = func(self)
            self.__dict__[atrribute_name] = val
            return val
    return property(_wrapper)

class TrainingBase:
    def __init__(self, config=None, ddp_rank=0, ddp_world_size=1):
        self.config_input = config
        self.config = self.get_default_config()
        if config is not None:
            for k in config.keys():
                if not k in self.config:
                    raise KeyError(f'Unknown config "{k}"')
            self.config.update(config)
        
        self.state = self.get_default_state()
        
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.comm = MPI.COMM_WORLD 
        self.is_distributed = (self.ddp_world_size > 1)
        self.is_main_rank = (self.ddp_rank == 0)
        
        if tlx.BACKEND == 'torch':
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler() if self.config.get('mixed_precision', False) else None
        elif tlx.BACKEND == 'tensorflow':
            from tensorflow.keras import mixed_precision
            if config.get('mixed_precision', False):  
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                self.scaler = True
        else:
            self.scaler = None
    
    @cached_property
    def train_dataset(self):
        raise NotImplementedError
    
    @cached_property
    def val_dataset(self):
        raise NotImplementedError
    
    @cached_property
    def collate_fn(self):
        return None

                                   
    @cached_property
    def train_sampler(self):
        import torch
        return  torch.utils.data.DistributedSampler(self.train_dataset,
                                                    shuffle=True)
    
    @cached_property
    def train_dataloader(self):
            
        common_kwargs = dict(
            dataset=self.train_dataset, 
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
        )

        if tlx.BACKEND == 'torch' and self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
            )
        else :
            common_kwargs['num_workers'] = self.config.dataloader_workers if self.config.dataloader_workers > 0 else 0
        
        if not self.is_distributed:
            dataloader = DataLoader(**common_kwargs, shuffle=True,
                                    drop_last=False)
        else:
            dataloader = DataLoader(**common_kwargs, 
                                    sampler=self.train_sampler)
        return dataloader
    
    @cached_property
    def val_dataloader(self):
        
            
        common_kwargs = dict(
            dataset=self.val_dataset, 
            collate_fn=self.collate_fn,
        )

        prediction_batch_size = self.config.batch_size * self.config.prediction_bmult
        common_kwargs['batch_size'] = prediction_batch_size
        common_kwargs['shuffle'] = False
        common_kwargs['drop_last'] = False

        if self.config.dataloader_workers > 0:
            common_kwargs.update(
                num_workers=self.config.dataloader_workers,
                persistent_workers=True,
            )
            

        # 分布式处理
        if self.is_distributed:
            sampler = DistributedTestDataSampler(
                data_source=self.val_dataset,
                batch_size=prediction_batch_size,
                rank=self.ddp_rank,
                world_size=self.ddp_world_size
            )
            return DataLoader(**common_kwargs, batch_sampler=sampler)
        else:
            return DataLoader(**common_kwargs)

    @cached_property
    def base_model(self):
        raise NotImplementedError
    
    
    @cached_property
    def model(self):
        model = self.base_model
        backend = tlx.BACKEND
        device = tlx.get_device()
        if tlx.BACKEND == 'torch':
            device_str = f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda'
            model = model.to(device_str)
        if self.is_distributed:
            if backend == 'torch':
                import torch
                model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[self.ddp_rank],
                                                              output_device=self.ddp_rank)
            elif backend == 'tensorflow':
                import tensorflow as tf
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    model = self.base_model.__class__(**self.get_model_config()[0])
        return model
    
    @cached_property
    def optimizer(self):
        config = self.config
        if tlx.BACKEND == 'tensorflow':
            import tensorflow as tf
            optimizer_class = getattr(tf.keras.optimizers, config.optimizer)
        else:
            import torch.optim as optim
            optimizer_class = getattr(optim, config.optimizer)
        
        optimizer_params = dict(
            lr=config.max_lr,
            **config.optimizer_params
        )
        
        if tlx.BACKEND == 'torch':
            # 获取原生PyTorch模型的参数
            params = self.model.parameters() 
            if self.is_distributed:
                params = self.model.module.parameters()
            optimizer_params['params'] = params
        
        base_optimizer = optimizer_class(**optimizer_params)
        
        # 混合精度处理
        if tlx.BACKEND == 'tensorflow' and config.mixed_precision:
            import tensorflow as tf
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        else:
            optimizer = base_optimizer
        
        return optimizer

    def get_default_config(self):
        return HDict(
            scheme=None,
            model_name='unnamed_model',
            distributed=False,
            random_seed=None,
            num_epochs=100,
            save_path=HDict.L('c:path.join("models")'),
            checkpoint_path=HDict.L('c:path.join(c.save_path,"checkpoint")'),
            config_path=HDict.L('c:path.join(c.save_path,"config")'),
            summary_path=HDict.L('c:path.join(c.save_path,"summary")'),
            log_path=HDict.L('c:path.join(c.save_path,"logs")'),
            validation_frequency=1,
            batch_size=HDict.L('c:128 if c.distributed else 32'),
            optimizer='Adam',
            max_lr=5e-4,
            clip_grad_value=None,
            optimizer_params={},
            dataloader_workers=0,
            dataloader_mp_context='forkserver',
            training_type='normal',
            evaluation_type='validation',
            predictions_path=HDict.L('c:path.join(c.save_path,"predictions")'),
            grad_accum_steps=1,
            prediction_bmult=1,
            mixed_precision = False,
            attn_threshold = 0.1,
            use_adaptive_sparse = False,  # 控制是否启用自适应稀疏
            sparse_alpha = 0.5 ,
        )


    def get_default_state(self):
        state = HDict(
            current_epoch = 0,
            global_step = 0,
        )
        return state


    def config_summary(self):
        if not self.is_main_rank: return
        for k, v in self.config.get_dict().items():
            print(f'{k} : {v}', flush=True)


    def save_config_file(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.config_path), exist_ok=True)
        save_config_to_file(self.config.get_dict(), self.config.config_path + '.yaml')
        save_config_to_file(self.config_input, self.config.config_path + '_input.yaml')


    def model_summary(self):
        if not self.is_main_rank: return
        os.makedirs(os.path.dirname(self.config.summary_path), exist_ok=True)
        trainable_params = 0
        non_trainable_params = 0
        
        if tlx.BACKEND == 'torch' and self.is_distributed:
            params = self.model.module.parameters() 
        else:
            params = self.model.trainable_weights
        
        for p in params:
            if tlx.BACKEND == 'torch':
                if p.requires_grad:
                    trainable_params += p.numel()
                else:
                    non_trainable_params += p.numel()
            else:
                if p.trainable:  
                    if tlx.BACKEND == 'tensorflow':
                        trainable_params += p.shape.num_elements()
                    else:
                        trainable_params += p.size
                else:
                    if tlx.BACKEND == 'tensorflow':
                        non_trainable_params += p.shape.num_elements()
                    else:
                        non_trainable_params += p.size
                    
        summary = dict(
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            model_representation=repr(self.model),
        )
        with open(self.config.summary_path + '.txt', 'w') as fp:
            yaml.dump(summary, fp, sort_keys=False, Dumper=yaml_Dumper)


    def save_checkpoint(self):
        if not self.is_main_rank:
            return
        ckpt_path = self.config.checkpoint_path
        os.makedirs(ckpt_path, exist_ok=True)

        state_list = []
        for key, value in self.state.items():
            state_list.append(tlx.convert_to_tensor(value))
        tlx.files.save_npz(save_list=state_list, name=os.path.join(ckpt_path, 'training_state.npz'))

        model_state_list = []
        if tlx.BACKEND == 'torch' and self.is_distributed:
            params = self.model.module.parameters()  
        else:
            params = self.model.trainable_weights
            
        for param in params:
            model_state_list.append(param)
        tlx.files.save_npz(save_list=model_state_list, name=os.path.join(ckpt_path, 'model_state.npz'))
        
        optimizer_state_list = []
        if tlx.BACKEND == 'tensorflow':
            optimizer_states = self.optimizer.get_weights()
            optimizer_state_list = [tlx.convert_to_tensor(state) for state in optimizer_states]
            tlx.files.save_npz(save_list=optimizer_state_list, name=os.path.join(ckpt_path, 'optimizer_state.npz'))
        elif tlx.BACKEND == 'torch':
            import torch
            torch.save(self.optimizer.state_dict(), os.path.join(ckpt_path, 'optimizer_state.pt'))
        
        """optimizer_state_list = []
        if tlx.BACKEND == 'tensorflow':
            optimizer_states = self.optimizer.get_weights()
        elif tlx.BACKEND == 'torch':
            optimizer_states = self.optimizer.state_dict().values()
        else:
            optimizer_states = []
        
        for state in optimizer_states:
            optimizer_state_list.append(tlx.convert_to_tensor(state))
        
        tlx.files.save_npz(save_list=optimizer_state_list, name=os.path.join(ckpt_path, 'optimizer_state.npz'))"""

        print(f'Checkpoint saved to: {ckpt_path}', flush=True)


    def load_checkpoint(self):
        ckpt_path = self.config.checkpoint_path
        try:
            state_npz = tlx.files.load_npz(name=os.path.join(ckpt_path, 'training_state.npz'))
            for key, value in zip(self.state.keys(), state_npz):
                self.state[key] = value

            model_state_npz = tlx.files.load_npz(name=os.path.join(ckpt_path, 'model_state.npz'))
            for i, param in enumerate(self.base_model.trainable_weights):
                if tlx.BACKEND == 'tensorflow':
                    param.assign(model_state_npz[i])
                elif tlx.BACKEND == 'torch':
                    param.data = tlx.convert_to_tensor(model_state_npz[i]).to(param.device)
                else:
                    raise NotImplementedError(f"Unsupported backend: {tlx.BACKEND}")

            """optimizer_state_npz = tlx.files.load_npz(name=os.path.join(ckpt_path, 'optimizer_state.npz'))
            if tlx.BACKEND == 'tensorflow' :
                optimizer_states = self.optimizer.set_weights(optimizer_states)
            else:
                optimizer_states = self.optimizer.load_state_dict(optimizer_states)"""
            
            if tlx.BACKEND == 'tensorflow':
                optimizer_state_npz = tlx.files.load_npz(name=os.path.join(ckpt_path, 'optimizer_state.npz'))
                self.optimizer.set_weights(optimizer_state_npz)
            elif tlx.BACKEND == 'torch':
                # 加载PyTorch优化器状态
                import torch
                optimizer_state = torch.load(os.path.join(ckpt_path, 'optimizer_state.pt'))
                self.optimizer.load_state_dict(optimizer_state)
                
            for i, state in enumerate(optimizer_states):
                state.assign(optimizer_state_npz[i])

            if self.is_main_rank:
                print(f'Checkpoint loaded from: {ckpt_path}', flush=True)
            
            if tlx.BACKEND == 'torch':
                import torch
                torch.cuda.empty_cache()
            

        except FileNotFoundError:
            pass


    # Callbacks
    def on_train_begin(self):
        pass


    def on_train_end(self):
        pass


    def on_epoch_begin(self, logs, training):
        pass


    def on_epoch_end(self, logs, training):
        pass


    def on_batch_begin(self, i, logs, training):
        pass


    def on_batch_end(self, i, logs, training):
        pass


    # Logging
    def get_verbose_logs(self):
        return OrderedDict(loss='0.4f')


    @cached_property
    def verbose_logs(self):
        return self.get_verbose_logs()


    def update_logs(self, logs, training, **updates):
        if training:
            logs.update(updates)
        else:
            logs.update(('val_' + k, v) for k, v in updates.items())


    def log_description(self, i, logs, training):
        if training:
            return list(f'{k} = {logs[k]:{f}}'
                        for k, f in self.verbose_logs.items())
        else:
            return list(f'val_{k} = {logs["val_" + k]:{f}}'
                        for k, f in self.verbose_logs.items())

        
    def preprocess_batch(self, batch):
        if isinstance(batch, CollatedBatch):
            return CollatedBatch(self.preprocess_batch(b) for b in batch)
        
        elif tlx.is_tensor(batch):
            if tlx.BACKEND == 'torch':
                device_str = f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda'
                return batch.to(device_str)
            elif tlx.BACKEND == 'tensorflow':
                return tlx.convert_to_tensor(tlx.convert_to_numpy(batch))
        elif isinstance(batch, dict):
            if tlx.BACKEND == 'torch':
                device_str = f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda'
                return batch.__class__(
                    (k, self.preprocess_batch(v).to(device_str)) for k, v in batch.items()
                )
            else:
                return batch.__class__(
                    (k, self.preprocess_batch(v)) for k, v in batch.items()
                )
        elif hasattr(batch, 'items'):
            device_str = f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda' if tlx.BACKEND == 'torch' else None
            return batch.__class__(
                (k, self.preprocess_batch(v).to(device_str))  # 添加设备转换
                for k, v in batch.items()
            )
            
        elif hasattr(batch, '__iter__'):
            device_str = f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda' if tlx.BACKEND == 'torch' else None
            return batch.__class__(
                self.preprocess_batch(v).to(device_str)  # 添加设备转换
                for v in batch
            )
        
        else:
            raise ValueError(f'Unsupported batch type: {type(batch)}')


    def calculate_loss(self, outputs, inputs):
        raise NotImplementedError


    def grad_accum_gather_outputs(self, outputs):
        return tlx.concat(outputs, dim=0)


    def grad_accum_reduce_loss(self, loss):
        total_loss = tlx.ops.sum(loss)
        return total_loss


    def grad_accum_collator(self, dataloader):
        dataloader_iter = iter(dataloader)
        if self.config.grad_accum_steps == 1:
            yield from dataloader_iter
        else:
            while True:
                collated_batch = CollatedBatch()
                try:
                    for _ in range(self.config.grad_accum_steps):
                        collated_batch.append(next(dataloader_iter))
                except StopIteration:
                    break
                finally:
                    if len(collated_batch) > 0: yield collated_batch

    @cached_property
    def train_steps_per_epoch(self):
        if self.config.grad_accum_steps == 1:
            return len(self.train_dataloader)
        else:
            return (len(self.train_dataloader) + self.config.grad_accum_steps - 1) // self.config.grad_accum_steps


    @cached_property
    def validation_steps_per_epoch(self):
        return len(self.val_dataloader)


    def training_step(self, batch, logs):
        for param in self.model.trainable_weights:
            if tlx.BACKEND == 'torch' and hasattr(param, 'grad'):
                param.grad = None  
            elif tlx.BACKEND == 'tensorflow' and hasattr(param, '_grad'):
                param._grad = None  
        
        # 混合精度上下文管理
        if tlx.BACKEND == 'torch':
            from torch.cuda.amp import autocast
            amp_ctx = autocast(enabled=self.config.mixed_precision)
        elif tlx.BACKEND == 'tensorflow':
            import tensorflow as tf
            amp_ctx = nullcontext()
        else:
            amp_ctx = nullcontext()
                
        with amp_ctx:
            if not isinstance(batch, CollatedBatch):
                outputs = self.model(batch)
                loss = self.calculate_loss(outputs=outputs, inputs=batch)
                
                if tlx.BACKEND == 'torch' and self.config.mixed_precision:
                    loss = loss.to(device=f'cuda:{self.ddp_rank}')  # 显式指定GPU设备
            else:
                num_nested_batches = len(batch)
                outputs = CollatedBatch()
                loss = CollatedBatch()
                
                sync_context = self.model.no_sync() if self.is_distributed else nullcontext()
                with sync_context:
                    for b in batch:
                        o = self.model(b)
                        l = self.calculate_loss(outputs=o, inputs=b) / num_nested_batches
                        
                        if tlx.BACKEND == 'torch' and self.config.mixed_precision:
                            l = l.to(device=f'cuda:{self.ddp_rank}')
                        
                        loss.append(l)
                        outputs.append(o)

        if tlx.BACKEND == 'torch':
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        elif tlx.BACKEND == 'tensorflow':
            import tensorflow as tf
            with tf.GradientTape() as tape:
                outputs = self.model(batch)
                loss = self.calculate_loss(outputs=outputs, inputs=batch)
            
            
            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        if self.config.clip_grad_value is not None:
            if tlx.BACKEND == 'torch' and self.scaler:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_value_(self.model.parameters(), self.config.clip_grad_value)

        # 参数更新
        if tlx.BACKEND == 'torch':
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        elif tlx.BACKEND == 'tensorflow':
            pass  

        return outputs, loss


    def validation_step(self, batch, logs):
        outputs = self.model(batch)
        loss = self.calculate_loss(outputs=outputs, inputs=batch)
        return outputs, loss


    def initialize_metrics(self, logs, training):
        pass


    def update_metrics(self, outputs, inputs, logs, training):
        pass


    def initialize_losses(self, logs, training):
        self._total_loss = 0.

    
    def update_losses(self, i, loss, inputs, logs, training):
        if not self.is_distributed:
            step_loss = tlx.ops.convert_to_numpy(loss).item()
        else:
            if training:
                loss = loss.detach()
            loss_np = tlx.ops.convert_to_numpy(loss)
            reduced_loss_np = tlx.ops.convert_to_numpy(tlx.ops.zeros_like(loss))
            self.comm.Allreduce(loss_np, reduced_loss_np, op=MPI.SUM)
            reduced_loss = tlx.ops.convert_to_tensor(reduced_loss_np)
            step_loss = reduced_loss.item() / self.ddp_world_size
        self._total_loss += step_loss
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss / (i + 1))


    def train_epoch(self, epoch, logs):
        if tlx.BACKEND == 'torch':
            self.model.train()
        else:
            self.model.set_train()
        
        self.initialize_losses(logs, True)
        self.initialize_metrics(logs, True)

        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)

        gen = self.grad_accum_collator(self.train_dataloader)
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True,
                    total=self.train_steps_per_epoch)
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch)
                outputs, loss = self.training_step(batch, logs)

                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)

                self.update_losses(i, loss, batch, logs, True)
                self.update_metrics(outputs, batch, logs, True)

                self.on_batch_end(i, logs, True)

                if self.is_main_rank:
                    desc = 'Training: ' + '; '.join(self.log_description(i, logs, True))
                    gen.set_description(desc)
        finally:
            if self.is_main_rank: gen.close()
            
            
            if tlx.BACKEND == 'torch' and self.is_distributed:
                params = self.model.module.parameters()
            else:
                params = self.model.trainable_weights
                
            for param in params:
                if hasattr(param, '_grad'): 
                    param._grad = None
                elif hasattr(param, 'grad'):
                    param.grad = None


    def minimal_train_epoch(self, epoch, logs):
        self.model.set_train()

        if self.is_distributed:
            self.train_sampler.set_epoch(epoch)

        gen = self.grad_accum_collator(self.train_dataloader)
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True, desc='Training: ',
                    total=self.train_steps_per_epoch)
        try:
            for i, batch in enumerate(gen):
                self.on_batch_begin(i, logs, True)
                batch = self.preprocess_batch(batch)
                _ = self.training_step(batch, logs)

                self.state.global_step = self.state.global_step + 1
                logs.update(global_step=self.state.global_step)

                self.on_batch_end(i, logs, True)
        finally:
            if self.is_main_rank: gen.close()
            for param in self.model.trainable_weights:
                if hasattr(param, '_grad'): 
                    param._grad = None
                elif hasattr(param, 'grad'):
                    param.grad = None


    def validation_epoch(self, epoch, logs):
        if tlx.BACKEND == 'tensorflow':
            self.model.trainable = False  
        else:
            self.model.eval()  
        
        self.initialize_losses(logs, False)
        self.initialize_metrics(logs, False)

        gen = self.val_dataloader
        if self.is_main_rank:
            gen = tqdm(gen, dynamic_ncols=True,
                    total=self.validation_steps_per_epoch)
        try:
            if tlx.BACKEND == 'torch':
                import torch
                grad_ctx = torch.no_grad()
            else:
                grad_ctx = nullcontext()
            
            with grad_ctx:
                for i, batch in enumerate(gen):
                    self.on_batch_begin(i, logs, False)
                    batch = self.preprocess_batch(batch)
                    outputs, loss = self.validation_step(batch, logs)

                    if tlx.BACKEND == 'tensorflow':
                        loss = tlx.convert_to_numpy(loss)
                    elif tlx.BACKEND == 'torch':
                        loss = loss.detach().cpu().numpy()

                    self.update_losses(i, loss, batch, logs, False)
                    self.update_metrics(outputs, batch, logs, False)

                    self.on_batch_end(i, logs, False)

                    if self.is_main_rank:
                        desc = 'Validation: ' + '; '.join(self.log_description(i, logs, False))
                        gen.set_description(desc)
        finally:
            if self.is_main_rank: 
                gen.close()
            if tlx.BACKEND == 'tensorflow':
                self.model.trainable = True
            else:
                self.model.train()


    def load_history(self):
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        try:
            with open(history_file, 'r') as fp:
                return yaml.load(fp, Loader=yaml_Loader)
        except FileNotFoundError:
            return []

    def save_history(self, history):
        os.makedirs(self.config.log_path, exist_ok=True)
        history_file = os.path.join(self.config.log_path, 'history.yaml')
        with open(history_file, 'w') as fp:
            yaml.dump(history, fp, sort_keys=False, Dumper=yaml_Dumper)


    def train_model(self):
        if self.is_main_rank:
            history = self.load_history()
        starting_epoch = self.state.current_epoch

        self.on_train_begin()
        should_stop_training = False
        try:
            for i in range(starting_epoch, self.config.num_epochs):
                self.state.current_epoch = i
                if self.is_main_rank:
                    print(f'\nEpoch {i + 1}/{self.config.num_epochs}:', flush=True)
                logs = dict(epoch=self.state.current_epoch,
                            global_step=self.state.global_step)

                try:
                    self.on_epoch_begin(logs, True)
                    if self.config.training_type == 'normal':
                        self.train_epoch(i, logs)
                    elif self.config.training_type == 'minimal':
                        self.minimal_train_epoch(i, logs)
                    else:
                        raise ValueError(f'Unknown training type: {self.config.training_type}')
                    self.on_epoch_end(logs, True)
                except StopTrainingException:
                    should_stop_training = True

                try:
                    if (self.val_dataloader is not None) \
                            and (not ((i + 1) % self.config.validation_frequency)):
                        self.on_epoch_begin(logs, False)
                        if self.config.evaluation_type == 'validation':
                            self.validation_epoch(i, logs)
                        elif self.config.evaluation_type == 'prediction':
                            self.prediction_epoch(i, logs)
                        else:
                            raise ValueError(f'Unknown evaluation type: {self.config.evaluation_type}')
                        self.on_epoch_end(logs, False)
                except StopTrainingException:
                    should_stop_training = True

                self.state.current_epoch = i + 1
                if self.is_main_rank:
                    self.save_checkpoint()

                    history.append(logs)
                    self.save_history(history)

                if should_stop_training:
                    if self.is_main_rank:
                        print('Stopping training ...')
                    break
        finally:
            self.on_train_end()

    
    def distributed_barrier(self):
        if self.is_distributed:
            dummy = tlx.ops.ones((), dtype=tlx.int64)
            dummy_np = tlx.ops.convert_to_numpy(dummy)
            reduced_dummy_np = tlx.ops.convert_to_numpy(tlx.ops.zeros_like(dummy))
            self.comm.Allreduce(dummy_np, reduced_dummy_np, op=MPI.SUM)
            reduced_dummy = tlx.ops.convert_to_tensor(reduced_dummy_np)


    def prediction_step(self, batch):
        predictions = self.model(batch)
        if isinstance(batch, tlx.Tensor):  
            return dict(inputs=batch, predictions=predictions)
        elif isinstance(batch, list):
            outputs = batch.copy()
            batch.append(predictions)
            return outputs
        elif isinstance(batch, dict):
            outputs = batch.copy()
            outputs.update(predictions=predictions)
            return outputs


    def prediction_loop(self, dataloader):
        self.model.set_eval()

        outputs = []

        if self.is_main_rank:
            gen = tqdm(dataloader, dynamic_ncols=True)
        else:
            gen = dataloader
        try:
            if tlx.BACKEND == 'torch':
                import torch
                grad_ctx = torch.no_grad()
            else:
                grad_ctx = nullcontext()
            
            with grad_ctx:
                for batch in gen:
                    batch = self.preprocess_batch(batch)
                    outputs.append(self.prediction_step(batch))
        finally:
            if self.is_main_rank:
                gen.close()

        return outputs
        
    def preprocess_predictions(self, outputs):
        if len(outputs) == 0:
            return outputs
        
        if tlx.ops.is_tensor(outputs[0]):
            return tlx.concat(outputs, axis=0)
        elif isinstance(outputs[0], dict):
            return {k: tlx.concat([o[k] for o in outputs], axis=0)
                    for k in outputs[0].keys()}
        elif isinstance(outputs[0], list):
            return [tlx.concat([o[i] for o in outputs], axis=0)
                    for i in range(len(outputs[0]))]
        else:
            raise ValueError('Unsupported output type')


    def postprocess_predictions(self, outputs):
        if tlx.ops.is_tensor(outputs):
            return tlx.convert_to_numpy(outputs)
        elif isinstance(outputs, dict):
            return {k: tlx.convert_to_numpy(v) for k, v in outputs.items()}
        elif isinstance(outputs, list):
            return [tlx.convert_to_numpy(v) for v in outputs]
        else:
            raise ValueError('Unsupported output type')

    def distributed_gather_tensor(self, tensors):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        shapes = tlx.ops.zeros(size + 1, dtype=tlx.int64)
        shapes_np = tlx.ops.convert_to_numpy(shapes)
        shapes_np[rank + 1] = tensors.shape[0]
        shapes = tlx.ops.convert_to_tensor(shapes_np)

        shapes_np = tlx.ops.convert_to_numpy(shapes)
        reduced_shapes_np = np.zeros_like(shapes_np)
        comm.Allreduce(shapes_np, reduced_shapes_np, op=MPI.SUM)
        reduced_shapes = tlx.ops.convert_to_tensor(reduced_shapes_np)

        offsets = tlx.ops.cumsum(reduced_shapes, axis=0)
        
        all_tensors_shape = (int(tlx.ops.convert_to_numpy(offsets[-1])), *tensors.shape[1:])
        all_tensors = tlx.ops.zeros(all_tensors_shape, dtype=tensors.dtype)

        start = int(tlx.ops.convert_to_numpy(offsets[rank]))
        end = int(tlx.ops.convert_to_numpy(offsets[rank + 1]))
        all_tensors[start:end] = tensors

        all_tensors_np = tlx.ops.convert_to_numpy(all_tensors)
        reduced_all_tensors_np = np.zeros_like(all_tensors_np)
        comm.Allreduce(all_tensors_np, reduced_all_tensors_np, op=MPI.SUM)
        reduced_all_tensors = tlx.ops.convert_to_tensor(reduced_all_tensors_np)

        return reduced_all_tensors


    def distributed_gather_predictions(self, predictions):
        if self.is_main_rank:
            print('Gathering predictions from all ranks...')

        if isinstance(predictions, tlx.Tensor):  
            all_predictions = self.distributed_gatther_tensor(predictions)
        elif isinstance(predictions, list):
            all_predictions = [self.distributed_gatther_tensor(pred) for pred in predictions]
        elif isinstance(predictions, dict):
            all_predictions = {key: self.distributed_gatther_tensor(pred)
                            for key, pred in predictions.items()}
        else:
            raise ValueError('Unsupported output type')

        if self.is_main_rank:
            print('Done.')
        return all_predictions


    def save_predictions(self, dataset_name, predictions):
        os.makedirs(self.config.predictions_path, exist_ok=True)
        predictions_file = os.path.join(self.config.predictions_path, f'{dataset_name}.npy')
        tlx.files.save_any_to_npy(predictions, predictions_file)
        print(f'Saved predictions to {predictions_file}')



    def evaluate_predictions(self, predictions):
        raise NotImplementedError


    def prediction_epoch(self, epoch, logs):
        if self.is_main_rank:
            print(f'Predicting on validation dataset...')
        dataloader = self.val_dataloader
        outputs = self.prediction_loop(dataloader)
        outputs = self.preprocess_predictions(outputs)

        if self.is_distributed:
            outputs = self.distributed_gather_predictions(outputs)

        predictions = self.postprocess_predictions(outputs)
        if self.is_main_rank:
            self.save_predictions('validation', predictions)
        results = self.evaluate_predictions(predictions)
        results = {f'val_{k}': v for k, v in results.items()}
        logs.update(results)
        if self.is_main_rank:
            desc = 'Validation: ' + '; '.join(f'{k}: {v:.4f}' for k, v in results.items())
            print(desc, flush=True)


    # Interface
    def prepare_for_training(self):
        self.config_summary()
        self.save_config_file()
        self.load_checkpoint()
        self.model_summary()


    def execute_training(self):
        self.prepare_for_training()
        self.train_model()
        self.finalize_training()


    def finalize_training(self):
        pass
    