from .training import TrainingBase, cached_property, CollatedBatch
from .testing import TestingBase
from .training_mixins import SaveModel, VerboseLR
from contextlib import nullcontext
from gammagl.utils.dotdict import HDict
import tensorlayerx as tlx
import numpy as np
from gammagl.data.graph_dataset import graphdata_collate

class EGTTraining(TestingBase, TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            model_name          = 'egt',
            cache_dir           = 'cache_data',
            dataset_name        = 'unnamed_dataset',
            dataset_path        = HDict.L('c:f"{c.cache_dir}/{c.dataset_name.upper()}"'),
            save_path           = HDict.L('c:path.join(f"models/{c.dataset_name.lower()}",c.model_name)'),
            model_height        = 4,
            node_width          = 64,
            edge_width          = 64,
            num_heads           = 8,
            node_dropout        = 0.,
            edge_dropout        = 0.,
            node_ffn_dropout    = HDict.L('c:c.node_dropout'),
            edge_ffn_dropout    = HDict.L('c:c.edge_dropout'),
            attn_dropout        = 0.,
            attn_maskout        = 0.,
            activation          = 'elu',
            clip_logits_value   = [-5,5],
            scale_degree        = True,
            node_ffn_multiplier = 1.,
            edge_ffn_multiplier = 1.,
            allocate_max_batch  = True,
            scale_dot_product   = True,
            egt_simple          = False,
        )
        return config

    def get_dataset_config(self):
        config = self.config
        dataset_config = dict(
            dataset_path = config.dataset_path,
            cache_dir    = config.cache_dir,
        )
        return dataset_config, None

    def get_model_config(self):
        config = self.config
        model_config = dict(
            model_height        = config.model_height         ,
            node_width          = config.node_width           ,
            edge_width          = config.edge_width           ,
            num_heads           = config.num_heads            ,
            node_mha_dropout    = config.node_dropout         ,
            edge_mha_dropout    = config.edge_dropout         ,
            node_ffn_dropout    = config.node_ffn_dropout     ,
            edge_ffn_dropout    = config.edge_ffn_dropout     ,
            attn_dropout        = config.attn_dropout         ,
            attn_maskout        = config.attn_maskout         ,
            activation          = config.activation           ,
            clip_logits_value   = config.clip_logits_value    ,
            scale_degree        = config.scale_degree         ,
            node_ffn_multiplier = config.node_ffn_multiplier  ,
            edge_ffn_multiplier = config.edge_ffn_multiplier  ,
            scale_dot           = config.scale_dot_product    ,
            egt_simple          = config.egt_simple           ,
        )
        return model_config, None

    def _cache_dataset(self, dataset):
        if self.is_main_rank:
            dataset.cache()
        self.distributed_barrier()
        if not self.is_main_rank:
            dataset.cache(verbose=0)

    def _get_dataset(self, split):
        dataset_config, dataset_class = self.get_dataset_config()
        if dataset_class is None:
            raise NotImplementedError('Dataset class not specified')
        
        dataset = dataset_class(**dataset_config, split=split)
        self._cache_dataset(dataset)
        return dataset

    @cached_property
    def train_dataset(self):
        return self._get_dataset('training')

    @cached_property
    def val_dataset(self):
        return self._get_dataset('validation')

    @cached_property
    def test_dataset(self):
        return self._get_dataset('test')

    @property
    def collate_fn(self):
        return graphdata_collate

    @cached_property
    def base_model(self):
        model_config, model_class = self.get_model_config()
        if model_class is None:
            raise NotImplementedError
        model = model_class(**model_config)
        return model

    def prepare_for_training(self):
        # cache datasets in same order on all ranks
        if self.is_distributed:
            self.train_dataset
            self.val_dataset
        super().prepare_for_training()
        
        if tlx.BACKEND == 'torch':
            tlx.set_device(device='cuda', id=0)  # 设置默认GPU设备
            
        # GPU memory cache for biggest batch
        if self.config.allocate_max_batch:
            if self.is_main_rank: print('Allocating cache for max batch size...', flush=True)

            if tlx.BACKEND =='torch':
                import torch
                torch.cuda.empty_cache()
            else: pass


            if tlx.BACKEND == 'torch':
                self.model.train()
            else:
                self.model.set_train()
            max_batch = self.train_dataset.max_batch(self.config.batch_size, self.collate_fn)
            max_batch = self.preprocess_batch(max_batch)
            
            def to_device(data):
                if tlx.is_tensor(data):
                    return tlx.ops.to_device(data, device=f'cuda:{self.ddp_rank}' if self.is_distributed else 'cuda')
                elif isinstance(data, dict):
                    return {k: to_device(v) for k, v in data.items()}
                elif isinstance(data, (list, tuple)):
                    return type(data)(to_device(v) for v in data)
                return data
            
            if tlx.BACKEND == 'torch':
                max_batch = to_device(max_batch)
                

            outputs = self.model(max_batch)
            
            if tlx.BACKEND == 'torch' and self.is_distributed:
                params = self.model.module.parameters()  # DDP模式访问原始模型
            else:
                params = self.model.trainable_weights
            
            if tlx.BACKEND == 'tensorflow':
                import tensorflow as tf  
                with tf.GradientTape() as tape:  
                    loss = self.calculate_loss(outputs=outputs, inputs=max_batch)
                grads = tape.gradient(loss, self.model.trainable_weights)
            elif tlx.BACKEND == 'torch':
                loss = self.calculate_loss(outputs=outputs, inputs=max_batch)
                loss.backward()
                grads = [param.grad for param in params]
            else:
                raise NotImplementedError(f"Unsupported backend: {tlx.BACKEND}")
            
            for param in params:  
                param.grad = None
    
    

    def initialize_losses(self, logs, training):
        self._total_loss = 0.
        self._total_samples = 0.

    def update_losses(self, i, loss, inputs, logs, training):
        if not isinstance(inputs, CollatedBatch):
            step_samples = float(inputs['num_nodes'].shape[0])
        else:
            step_samples = float(sum(i['num_nodes'].shape[0] for i in inputs))
        
        if tlx.BACKEND == 'torch':
            """if isinstance(loss, (float, int)):
                loss_value = float(loss)
            else:
                loss_value = loss.detach().cpu().item()"""
            loss_value = float(loss)
        else:  
            if isinstance(loss, (np.ndarray, np.generic)):
                loss_value = float(loss)
            else:
                loss_value = float(tlx.ops.convert_to_numpy(loss))
            
        if not self.is_distributed:
            step_loss = loss_value * step_samples
        else:
            step_samples = tlx.convert_to_tensor(step_samples, dtype=loss.dtype)  

            if training:
                loss = loss.detach()
            step_loss = loss * step_samples

            if tlx.BACKEND == 'torch':
                import torch.distributed as dist
                dist.all_reduce(step_loss)
                dist.all_reduce(step_samples)
            elif tlx.BACKEND == 'tensorflow':
                import tensorflow as tf
                step_loss = tf.distribute.get_replica_context().all_reduce('sum', step_loss)
                step_samples = tf.distribute.get_replica_context().all_reduce('sum', step_samples)
            
            step_loss = step_loss.numpy()  
            step_samples = step_samples.numpy()  

        self._total_loss += step_loss
        self._total_samples += step_samples
        self.update_logs(logs=logs, training=training,
                         loss=self._total_loss/self._total_samples)