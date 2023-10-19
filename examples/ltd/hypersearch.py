import os
import psutil
from utils import choose_path, load_cascades
from distill_dgl import choose_model, model_train
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
from gammagl.datasets import Planetoid, Amazon


def raw_experiment(configs):
    print('*************************************')
    print(psutil.Process(os.getpid()).memory_info().rss)  # 了解内存占用情况
    output_dir, cascade_dir = choose_path(configs)
    # load datasets
    if str.lower(configs['dataset']) in ['cora', 'pubmed', 'citeseer']:
        dataset = Planetoid(configs['dataset_path'], configs['dataset'], split="random", num_train_per_class=50, num_val=0, num_test=2358)
        dataset.process()
        graph = dataset[0]
    elif str.lower(configs['dataset']) in ['computers', 'photo']:
        dataset = Amazon(configs['dataset_path'], configs['dataset'])
        dataset.process()
        graph = dataset[0]
    else:
        raise ValueError('Unknown dataset: {}'.format(configs['dataset']))
    # 获得教师模型的预测结果
    teacher_logits = load_cascades(cascade_dir)
    model = choose_model(configs, dataset)
    acc_val = model_train(configs, model, graph, teacher_logits, dataset)
