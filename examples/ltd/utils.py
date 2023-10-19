import yaml
from pathlib import Path
import numpy as np
import tensorlayerx as tlx
import os


def get_training_config(config_path, model_name):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    specific_config = dict(full_config['global'], **full_config[model_name])
    specific_config['model_name'] = model_name
    return specific_config


# def check_device(conf):
#     if conf['model_name'] in ['DeepWalk', 'GraphSAGE']:
#         is_cuda = False
#     else:
#         is_cuda = not conf['no_cuda'] and len(tf.config.list_physical_devices('GPU')) > 0
#     if is_cuda:
#         tf.random.set_seed(conf['seed'])
#     device = "/GPU:0" if is_cuda else "/CPU:0"
#     return device


def choose_path(conf):
    if 'assistant' not in conf.keys():
        teacher_str = conf['teacher']
    elif conf['assistant'] == 0:
        teacher_str = 'nasty_' + conf['teacher']
    elif conf['assistant'] == 1:
        teacher_str = 'reborn_' + conf['teacher']
    else:
        raise ValueError(r'No such assistant')
    output_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str + '_' + conf['student'],
                                     'cascade_random_' + str(conf['division_seed']))
    cascade_dir = Path.cwd().joinpath('outputs', conf['dataset'], teacher_str,
                                      'cascade_random_' + str(conf['division_seed']) + '_' + str(conf['labelrate']),
                                      'logits.npy')
    return output_dir, cascade_dir


# def set_random_seed(seed):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     tf.config.threading.set_inter_op_parallelism_threads(1)


def load_cascades(cascade_dir):
    # 从文件中加载 NumPy 数组
    loaded_logits_array = np.load(cascade_dir)

    # 创建 tensorlayerx 张量
    loaded_logits = tlx.ops.convert_to_tensor(loaded_logits_array)
    return loaded_logits
