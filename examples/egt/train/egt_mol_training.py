import tensorlayerx as tlx
import numpy as np
from .egt_training import EGTTraining
from .training_mixins import LinearLRWarmupCosineDecay, VerboseLR

class EGT_MOL_Training(LinearLRWarmupCosineDecay, VerboseLR, EGTTraining):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            num_virtual_nodes       = 1,
            upto_hop                = 16,
            svd_calculated_dim      = 8,
            svd_output_dim          = 8,
            svd_random_neg          = True,
            pretrained_weights_file = None,
            num_epochs              = 1000,
        )
        return config_dict
    
    def get_dataset_config(self):
        dataset_config, dataset_class = super().get_dataset_config()
        if self.config.svd_output_dim > 0:
            dataset_config.update(
                calculated_dim    = self.config.svd_calculated_dim,
                output_dim        = self.config.svd_output_dim,
                random_neg_splits = ['training'] if self.config.svd_random_neg else [],
            )
        return dataset_config, dataset_class
    
    def get_model_config(self):
        model_config, model_class = super().get_model_config()
        model_config.update(
            num_virtual_nodes = self.config.num_virtual_nodes,
            upto_hop          = self.config.upto_hop,
            svd_encodings     = self.config.svd_output_dim,
        )
        return model_config, model_class
    
    def load_checkpoint(self):
        super().load_checkpoint()
        w_file = self.config.pretrained_weights_file
        if w_file is not None and self.state.global_step == 0:
            weights = np.load(w_file, allow_pickle=True)
            if isinstance(weights, np.ndarray):
                weights = weights.item()
            for k in list(weights.keys()).copy():
                if 'mlp_layers.2' in k:
                    del weights[k]
                        
            missing, unexpected = self.base_model.load_dict(weights, strict=False)
            backend = tlx.get_backend()
            if backend =='torch':
                import torch
                torch.cuda.empty_cache()
            else :pass

            if self.is_main_rank:
                print(f'Loaded pretrained weights from {w_file}',flush=True)
                print(f'missing keys: {missing}',flush=True)
                print(f'unexpected keys: {unexpected}',flush=True)