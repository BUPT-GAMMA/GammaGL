import os
from pathlib import Path
import numpy as np
import tensorlayerx as tlx
import logging

# 获取 TensorLayerX 的 logger
tlx_logger = logging.getLogger("tensorlayerx")

# 设置 logger 的级别为 ERROR，以屏蔽 INFO 信息
tlx_logger.setLevel(logging.ERROR)
from distill_dgl import choose_model, model_train, layer_normalize
import sys
import optuna
import torch

sys.path.insert(0, os.path.abspath('../../'))
from gammagl.datasets import Planetoid


def choose_path(conf):
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], conf['teacher'],
                                      'logits.npy')
    return cascade_dir


def load_cascades(cascade_dir):
    loaded_logits_array = np.load(cascade_dir)
    loaded_logits = tlx.ops.convert_to_tensor(loaded_logits_array)
    return loaded_logits


class AutoML(object):
    def __init__(self, kwargs, func_search):
        self.default_params = kwargs
        self.func_search = func_search
        self.n_trials = kwargs['ntrials']
        self.n_jobs = kwargs['njobs']
        self.best_acc_test = None
        self.mylr = None
        self.tlr = None

    def _objective(self, trials):
        params = self.default_params
        params.update(self.func_search(trials))
        lr = params['my_lr']
        tr = params['my_t_lr']
        acc_test = raw_experiment(params)
        if self.best_acc_test is None or acc_test > self.best_acc_test:
            self.best_acc_test = acc_test
            self.mylr = lr
            self.tlr = tr
        return acc_test

    def run(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials,
                       n_jobs=self.n_jobs)
        return self.best_acc_test, self.mylr, self.tlr


def raw_experiment(configs):
    cascade_dir = choose_path(configs)
    if str.lower(configs['dataset']) == 'cora':
        dataset = Planetoid(configs['dataset_path'], configs['dataset'], split="random", num_train_per_class=50,
                            num_val=0, num_test=2358)
        dataset.process()
        graph = dataset[0]
    elif str.lower(configs['dataset']) == 'pubmed':
        dataset = Planetoid(configs['dataset_path'], configs['dataset'], split="random", num_train_per_class=50,
                            num_val=0, num_test=19000)
        dataset.process()
        graph = dataset[0]
    elif str.lower(configs['dataset']) == 'citeseer':
        dataset = Planetoid(configs['dataset_path'], configs['dataset'], split="random", num_train_per_class=50,
                            num_val=0, num_test=3000)
        dataset.process()
        graph = dataset[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(configs['dataset']))
    teacher_logits = load_cascades(cascade_dir)
    # if configs['student'] == 'GAT' and configs['dataset'] == 'citeseer':
    teacher_logits = layer_normalize(teacher_logits)
    model = choose_model(configs, dataset)
    acc_val = model_train(configs, model, graph, teacher_logits, dataset)

    return acc_val
