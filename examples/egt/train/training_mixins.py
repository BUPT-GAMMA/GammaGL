
from .training import TrainingBase, StopTrainingException
from gammagl.utils.dotdict import HDict

import tensorlayerx as tlx  # 替换 torch 为 tensorlayerx
import numpy as np
import os


class SaveModel(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            saved_model_path           = HDict.L('c:path.join(c.save_path,"saved_model")'),
            save_model_when            = 'epoch',
            saved_model_name           = "epoch_{epoch:0>4d}",
            save_model_monitor         = 'val_loss',
            save_monitor_improves_when = 'less',
            save_model_condition       = HDict.L("c:c.save_model_monitor+"+
                                                 "('<=' if c.save_monitor_improves_when=='less' else '>=')+"+
                                                 "'save_monitor_value'"),
            save_last_only             = True,
        )
        return config
    
    def get_default_state(self):
        state = super().get_default_state()
        state.update(
            last_saved_model_file = None,
        )
        if self.config.save_monitor_improves_when == 'less':
            state.update(
                save_monitor_value = np.inf,
                save_monitor_epoch = -1,
            )
        elif self.config.save_monitor_improves_when == 'greater':
            state.update(
                save_monitor_value = 0,
                save_monitor_epoch = -1,
            )
        else:
            raise ValueError
        return state
    
    def save_model(self, name):
        if not self.is_main_rank: return
        
        os.makedirs(self.config.saved_model_path, exist_ok=True)
        save_file = os.path.join(self.config.saved_model_path, name+'.npz')  # 替换 .pt 为 .npz

        self.base_model.save_weights(save_file, format='npz_dict')  
        print(f'SAVE: model saved to {save_file}', flush=True)
        
        if self.config.save_last_only and (self.state.last_saved_model_file is not None)\
                                        and (os.path.exists(self.state.last_saved_model_file)):
            os.remove(self.state.last_saved_model_file)
            print(f'SAVE: removed old model file {self.state.last_saved_model_file}', flush=True)
        
        self.state.last_saved_model_file = save_file
    
    def on_batch_end(self, i, logs, training):
        super().on_batch_end(i, logs, training)
        if self.config.save_model_when != 'batch' or not training or not self.is_main_rank: return
        config = self.config
        scope = dict(batch=i)
        scope.update(self.state)
        scope.update(logs)
        if eval(config.save_model_condition, scope):
            self.save_model(config.saved_model_name.format(**scope))
    
    def on_epoch_end(self, logs, training):
        super().on_epoch_end(logs, training)
        if training: return
        
        config = self.config
        state = self.state
        monitor = config.save_model_monitor
        try:
            new_value = logs[monitor]
            new_epoch = logs['epoch']
        except KeyError:
            print(f'Warning: SAVE: COULD NOT FIND LOG!', flush=True)
            return
        
        old_value = state.save_monitor_value
        old_epoch = state.save_monitor_epoch
        
        if (self.config.save_monitor_improves_when == 'less' and new_value <= old_value)\
            or (self.config.save_monitor_improves_when == 'greater' and new_value >= old_value):
            state.save_monitor_value = new_value
            state.save_monitor_epoch = new_epoch
            if self.is_main_rank:
                print(f'MONITOR BEST: {monitor} improved from (epoch:{old_epoch},value:{old_value:0.5f})'+
                        f' to (epoch:{new_epoch},value:{new_value:0.5f})',flush=True)
        elif self.is_main_rank:
            print(f'MONITOR BEST: {monitor} did NOT improve from'+
                          f' (epoch:{old_epoch},value:{old_value:0.5f})',flush=True)
            
        if config.save_model_when != 'epoch' or not self.is_main_rank: return
        scope = {}
        scope.update(self.state)
        scope.update(logs)
        if eval(config.save_model_condition, scope):
            self.save_model(config.saved_model_name.format(**scope))


class VerboseLR(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            verbose_lr_log            = True,
        )
        return config
    
    def log_description(self, i, logs, training):
        descriptions = super().log_description(i, logs, training)
        if training and self.config.verbose_lr_log:
            descriptions.append(f'(lr:{logs["lr"]:0.3e})')
        return descriptions


class ReduceLR(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            rlr_factor                = 0.5,
            rlr_patience              = 10,
            min_lr                    = 1e-6,
            stopping_lr               = 0.,
            rlr_monitor               = 'val_loss',
            rlr_monitor_improves_when = 'less',
        )
        return config
    
    def get_default_state(self):
        state = super().get_default_state()
        state.update(
            last_rlr_epoch = -1,
        )
        if self.config.rlr_monitor_improves_when == 'less':
            state.update(
                rlr_monitor_value = np.inf,
                rlr_monitor_epoch = -1,
            )
        elif self.config.rlr_monitor_improves_when == 'greater':
            state.update(
                rlr_monitor_value = 0,
                rlr_monitor_epoch = -1,
            )
        else:
            raise ValueError
        return state
    
    def on_epoch_begin(self, logs, training):
        super().on_epoch_begin(logs, training)
        if 'lr' not in logs:
            logs['lr'] = self.optimizer.learning_rate  
            if callable(logs['lr']):
                logs['lr'] = logs['lr']()
    
    def on_epoch_end(self, logs, training):
        super().on_epoch_end(logs, training)
        if training: return
        
        config = self.config
        state = self.state
        monitor = config.rlr_monitor
        try:
            new_value = logs[monitor]
            new_epoch = logs['epoch']
        except KeyError:
            print(f'Warning: RLR: COULD NOT FIND LOG!', flush=True)
            return
        
        old_value = state.rlr_monitor_value
        old_epoch = state.rlr_monitor_epoch
        
        if (self.config.rlr_monitor_improves_when == 'less' and new_value <= old_value)\
            or (self.config.rlr_monitor_improves_when == 'greater' and new_value >= old_value):
            state.rlr_monitor_value = new_value
            state.rlr_monitor_epoch = new_epoch
        else:
            if config.rlr_factor < 1:
                epoch_gap = (new_epoch - max(state.last_rlr_epoch, old_epoch))
                if epoch_gap >= config.rlr_patience:
                    old_lrs = []
                    new_lrs = []
                    old_lr = self.optimizer.learning_rate  
                    if callable(old_lr):
                        old_lr = old_lr()
                    new_lr = max(old_lr * config.rlr_factor, config.min_lr)
                    self.optimizer.learning_rate = new_lr
                    old_lrs.append(old_lr)
                    new_lrs.append(new_lr)
                    
                    old_lr = max(old_lrs)
                    new_lr = max(new_lrs)
                    
                    logs['lr'] = new_lr
                    
                    state.last_rlr_epoch = new_epoch
                    if self.is_main_rank:
                        print(f'\nRLR: {monitor} did NOT improve for {epoch_gap} epochs,'+
                                  f' new lr = {new_lr}', flush=True)
                
                    if new_lr < config.stopping_lr:
                        if self.is_main_rank:
                            print(f'\nSTOP: lr fell below {config.stopping_lr}, STOPPING TRAINING!',flush=True)
                        raise StopTrainingException


class LinearLRWarmup(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            lr_warmup_steps = -1,
        )
        return config
    def on_batch_begin(self, i, logs, training):
        super().on_batch_begin(i, logs, training)
        if training and self.state.global_step <= self.config.lr_warmup_steps:
            new_lr = self.config.max_lr * (self.state.global_step / self.config.lr_warmup_steps)
            self.optimizer.learning_rate = new_lr  
            logs['lr'] = new_lr


class LinearLRWarmupCosineDecay(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            lr_warmup_steps = 60_000,
            lr_total_steps  = 1_000_000,
            min_lr          = 1e-6,
            cosine_halfwave = False,
        )
        return config
    def on_batch_begin(self, i, logs, training):
        super().on_batch_begin(i, logs, training)
        if training:
            global_step = self.state.global_step
            lr_total_steps = self.config.lr_total_steps
            lr_warmup_steps = self.config.lr_warmup_steps
            max_lr = self.config.max_lr
            min_lr = self.config.min_lr
            
            if global_step > lr_total_steps:
                if self.is_main_rank:
                    print(f'\nSTOP: global_step > lr_total_steps, STOPPING TRAINING!',flush=True)
                raise StopTrainingException
            
            if global_step <= lr_warmup_steps:
                new_lr = min_lr + (max_lr - min_lr) * (global_step / lr_warmup_steps)
            else:
                if self.config.cosine_halfwave:
                    new_lr = min_lr + (max_lr - min_lr) * np.cos(0.5 * np.pi * (global_step - lr_warmup_steps) / (lr_total_steps - lr_warmup_steps))
                else:
                    new_lr = min_lr + (max_lr - min_lr) * (1 + np.cos(np.pi * (global_step - lr_warmup_steps) / (lr_total_steps - lr_warmup_steps))) * 0.5
                new_lr = float(new_lr)
            

            self.optimizer.learning_rate = new_lr  
            logs['lr'] = new_lr