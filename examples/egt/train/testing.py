import tensorlayerx as tlx
from tensorlayerx.dataflow import DataLoader
import numpy as np
import os
import yaml
from yaml import SafeDumper as yaml_Dumper

from gammagl.utils.dotdict import HDict
from .training import TrainingBase, DistributedTestDataSampler, cached_property


class TestingBase(TrainingBase):
    def get_default_config(self):
        config = super().get_default_config()
        config.update(
            state_file         = None,
            predict_on         = ['train', 'val', 'test'],
            evaluate_on        = HDict.L('c:c.predict_on'),
            predictions_path   = HDict.L('c:path.join(c.save_path,"predictions")'),
        )
        return config
    
    @cached_property
    def test_dataset(self):
        raise NotImplementedError
    
    @cached_property
    def train_pred_dataloader(self):
        prediction_batch_size = self.config.batch_size*self.config.prediction_bmult
        if not self.is_distributed:
            dataloader = DataLoader(dataset=self.train_dataset,
                                    batch_size=prediction_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=self.collate_fn,
                                    )
        else:
            sampler = DistributedTestDataSampler(data_source=self.train_dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(dataset=self.train_dataset,
                                    collate_fn=self.collate_fn,
                                    batch_sampler=sampler,
                                    )
        return dataloader
    
    @cached_property
    def test_dataloader(self):
        prediction_batch_size = self.config.batch_size*self.config.prediction_bmult
        if not self.is_distributed:
            dataloader = DataLoader(dataset=self.test_dataset,
                                    batch_size=prediction_batch_size,
                                    shuffle=False,
                                    drop_last=False,
                                    collate_fn=self.collate_fn,
                                    )
        else:
            sampler = DistributedTestDataSampler(data_source=self.test_dataset,
                                                 batch_size=prediction_batch_size,
                                                 rank=self.ddp_rank,
                                                 world_size=self.ddp_world_size)
            dataloader = DataLoader(dataset=self.test_dataset,
                                    collate_fn=self.collate_fn,
                                    batch_sampler=sampler,
                                    )
        return dataloader
    
    
    def test_dataloader_for_dataset(self, dataset_name):
        if dataset_name == 'train':
            return self.train_pred_dataloader
        elif dataset_name == 'val':
            return self.val_dataloader
        elif dataset_name == 'test':
            return self.test_dataloader
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')
    
    def predict_and_save(self):
        for dataset_name in self.config.predict_on:
            if self.is_main_rank:
                print(f'Predicting on {dataset_name} dataset...')
            dataloader = self.test_dataloader_for_dataset(dataset_name)
            outputs = self.prediction_loop(dataloader)
            outputs = self.preprocess_predictions(outputs)
        
            if self.is_distributed:
                outputs = self.distributed_gather_predictions(outputs)
            
            if self.is_main_rank:
                predictions = self.postprocess_predictions(outputs)
                self.save_predictions(dataset_name, predictions)
    
    
    def load_model_state(self):
        if self.config.state_file is None:
            state_file = os.path.join(self.config.checkpoint_path, 'model_state.npz')
        else:
            state_file = self.config.state_file

        self.base_model.load_weights(state_file, format='npz_dict')
        
        if self.is_main_rank:
            print(f'Loaded model state from {state_file}')
        
    def prepare_for_testing(self):
        self.config_summary()
        self.load_model_state()
        
    def make_predictions(self):
        self.prepare_for_testing()
        self.predict_and_save()
        if len(self.config.evaluate_on) > 0:
            self.evaluate_and_save()
    
    
    def get_dataset(self, dataset_name):
        if dataset_name == 'train':
            return self.train_dataset
        elif dataset_name == 'val':
            return self.val_dataset
        elif dataset_name == 'test':
            return self.test_dataset
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')
    
    def evaluate_on(self, dataset_name, dataset, predictions):
        raise NotImplementedError()
    
    def evaluate_and_save(self):
        if not self.is_main_rank:
            return
        
        results = {}
        results_file = os.path.join(self.config.predictions_path, 'results.yaml')
        
        for dataset_name in self.config.evaluate_on:
            dataset = self.get_dataset(dataset_name)

            predictions = tlx.files.load_npy_to_any(path=self.config.predictions_path, name=f'{dataset_name}.npy') 
            dataset_results = self.evaluate_on(dataset_name, dataset, predictions)

            for k,v in dataset_results.items():
                print(f'{dataset_name} {k}: {v}')
            
            results[dataset_name] = dataset_results
            with open(results_file, 'w') as fp:
                yaml.dump(results, fp, sort_keys=False, Dumper=yaml_Dumper)      
    
    
    def do_evaluations(self):
        self.evaluate_and_save()
    
    def finalize_training(self):
        super().finalize_training()
        self.predict_and_save()
        if len(self.config.evaluate_on) > 0:
            self.evaluate_and_save()