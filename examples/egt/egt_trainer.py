import numpy as np
import tensorlayerx as tlx
from tensorlayerx import nn

from examples.egt.train.training import cached_property
from examples.egt.train.egt_mol_training import EGT_MOL_Training

from gammagl.models.egt import EGT_PCQM4MV2
from gammagl.datasets.pcqm4mv2 import PCQM4Mv2StructuralSVDGraphDataset


class PCQM4MV2_Training(EGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name = 'pcqm4mv2',
            dataset_path = 'cache_data/PCQM4MV2',
            predict_on   = ['val','test'],
            evaluate_on  = ['val','test'],
            state_file   = None,
        )
        return config_dict
    
    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, PCQM4Mv2StructuralSVDGraphDataset
    
    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, EGT_PCQM4MV2
    
    def calculate_loss(self, outputs, inputs):
        return tlx.losses.absolute_difference_error(outputs, inputs['target'])
    
    @cached_property
    def evaluator(self):
        from ogb.lsc.pcqm4mv2 import PCQM4Mv2Evaluator
        evaluator = PCQM4Mv2Evaluator()
        return evaluator
    
    def prediction_step(self, batch):
        return dict(
            predictions = self.model(batch),
            targets     = batch['target'],
        )
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        if dataset_name == 'test':
            self.evaluator.save_test_submission(
                input_dict = {'y_pred': predictions['predictions']},
                dir_path = self.config.predictions_path,
                mode = 'test-dev',
            )
            print(f'Saved final test-dev predictions to {self.config.predictions_path}')
            return {'mae': np.nan}
        
        print(f'Evaluating on {dataset_name}')
        input_dict = {"y_true": predictions['targets'], 
                      "y_pred": predictions['predictions']}
        results = self.evaluator.eval(input_dict)
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()
        return results

SCHEME = PCQM4MV2_Training

import sys
import logging
from examples.egt.train.execute import get_configs_from_args, execute
logging.getLogger('tensorlayerx').setLevel(logging.WARNING)  # 隐藏INFO级日志

if __name__ == '__main__':
    config = get_configs_from_args(sys.argv)
    execute('train', config)
    execute('predict', config)
    execute('evaluate', config)
