import os
os.environ['TL_BACKEND'] = 'torch'
import argparse
import itertools
from collections import defaultdict, namedtuple
from hypersearch import raw_experiment

from pathlib import Path
from utils import get_training_config
# from utils.logger import output_results
import tensorlayerx as tlx

predefined_configs = {
    'GCN': {
        'cora': {
            'num_layers': 8,
            'emb_dim': 32,  # 嵌入向量的维度
            'feat_drop': 0.8,  # 特征丢弃率
            'attn_drop': 0.2,  # 注意力丢弃率
            'beta': 0,  # 正则化参数
            'lr': 0.001,
            'wd': 0.01,  # 权重衰减参数
            'my_lr': 0.000192335844492524,
            'my_t_lr': 0.000259968648780918,
            'lam': 1,  # 正则化参数
            'k': 1  # 用于某些模型（如 GAT、APPNP、GraphSAGE、SGC）的邻居采样的参数
        },
        'citeseer': {
            'num_layers': 6,
            'emb_dim': 64,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 6.90460565438149E-06,
            'my_t_lr': 0.0000746422264911547,
            'lam': 1,
            'k': 1
        },
        'pubmed': {
            'num_layers': 5,
            'emb_dim': 8,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 7.39520400332059E-06,
            'my_t_lr': 0.0000937937968682644,
            'lam': 200,
            'k': 1
        },
        'amazon_electronics_computers': {
            'num_layers': 10,
            'emb_dim': 64,
            'feat_drop': 0.5,
            'attn_drop': 0.8,
            'beta': 1,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 3.84927458233582E-06,
            'my_t_lr': 0.000011810083910476,
            'lam': 100,
            'k': 1
        },
        'amazon_electronics_photo': {
            'num_layers': 6,
            'emb_dim': 64,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000101024021023117,
            'my_t_lr': 0.000188676574559862,
            'lam': 50,
            'k': 1
        },
    },
    'GAT': {
        'cora': {
            'num_layers': 5,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.00167558762679301,
            'my_t_lr': 0.000329262637088989,
            'lam': 1,
            'k': 2
        },
        'citeseer': {
            'num_layers': 5,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000624339224884044,
            'my_t_lr': 0.000170703192020892,
            'lam': 1,
            'k': 2
        },
        'pubmed': {
            'num_layers': 6,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000295072855477947,
            'my_t_lr': 0.000003722992284806,
            'lam': 200,
            'k': 2
        },
        'amazon_electronics_computers': {
            'num_layers': 10,
            'emb_dim': 64,
            'feat_drop': 0.2,
            'attn_drop': 0.5,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 3.56369930138321E-06,
            'my_t_lr': 0.0000681651427560019,
            'lam': 200,
            'k': 2
        },
        'amazon_electronics_photo': {
            'num_layers': 10,
            'emb_dim': 64,
            'feat_drop': 0.5,
            'attn_drop': 0.5,
            'beta': 1,
            'lr': 0.01,
            'wd': 0.01,
            'my_lr': 0.0000189160005829701,
            'my_t_lr': 0.000178003872136085,
            'lam': 50,
            'k': 2
        },
    },
    'APPNP': {
        'cora': {
            'num_layers': 5,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.000724886927941229,
            'my_t_lr': 0.000415013790497188,
            'lam': 1,
            'k': 3
        },
        'citeseer': {
            'num_layers': 7,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.00025999720049686,
            'my_t_lr': 0.000044216346318599,
            'lam': 1,
            'k': 3
        },
        'pubmed': {
            'num_layers': 8,
            'emb_dim': 64,
            'feat_drop': 0.5,
            'attn_drop': 0.2,
            'beta': 5,
            'lr': 0.005,
            'wd': 0.01,
            'my_lr': 4.6793268540066124e-05,
            'my_t_lr': 7.510813746227752e-05,
            'lam': 200,
            'k': 1
        },
        'amazon_electronics_computers': {
            'num_layers': 7,
            'emb_dim': 64,
            'feat_drop': 0.2,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.01,
            'wd': 0.0005,
            'my_lr': 8.59617764272168E-06,
            'my_t_lr': 0.0000142111993284188,
            'lam': 200,
            'k': 3
        },
        'amazon_electronics_photo': {
            'num_layers': 10,
            'emb_dim': 64,
            'feat_drop': 0.5,
            'attn_drop': 0.8,
            'beta': 0,
            'lr': 0.01,
            'wd': 0.01,
            'my_lr': 0.0000298839079307718,
            'my_t_lr': 5.64252783326123E-06,
            'lam': 50,
            'k': 3
        },
    },
    'GraphSAGE': {
        'cora': {
            'num_layers': 9,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0008736665883794245,
            'my_t_lr': 0.00022372258507504132,
            'lam': 1,
            'k': 3
        },
        'citeseer': {
            'num_layers': 7,
            'emb_dim': 16,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 9.78559574335461E-06,
            'my_t_lr': 0.000205531324864926,
            'lam': 1,
            'k': 3
        },
        'pubmed': {
            'num_layers': 9,
            'emb_dim': 64,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000557930446190816,
            'my_t_lr': 0.0004787315931149979,
            'lam': 100,
            'k': 3
        },
        'amazon_electronics_computers': {
            'num_layers': 8,
            'emb_dim': 64,
            'feat_drop': 0.2,
            'attn_drop': 0.5,
            'beta': 0,
            'lr': 0.01,
            'wd': 0.0005,
            'my_lr': 6.86132690032694E-07,
            'my_t_lr': 0.0000112049183440808,
            'lam': 200,
            'k': 3
        },
        'amazon_electronics_photo': {
            'num_layers': 9,
            'emb_dim': 64,
            'feat_drop': 0.8,
            'attn_drop': 0.8,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 7.19032603009029E-06,
            'my_t_lr': 0.000065970596042317,
            'lam': 50,
            'k': 3
        },
    },
    'SGC': {
        'cora': {
            'num_layers': 5,
            'emb_dim': 32,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0007135817301312403,
            'my_t_lr': 0.00019161535850566015,
            'lam': 1,
            'k': 2
        },
        'citeseer': {
            'num_layers': 10,
            'emb_dim': 16,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000123858253110345,
            'my_t_lr': 0.000135485817980452,
            'lam': 1,
            'k': 1
        },
        'pubmed': {
            'num_layers': 9,
            'emb_dim': 16,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 0.0000791924817557886,
            'my_t_lr': 0.000416428933461071,
            'lam': 100,
            'k': 1
        },
        'amazon_electronics_computers': {
            'num_layers': 8,
            'emb_dim': 64,
            'feat_drop': 0.5,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.0005,
            'my_lr': 5.99975517464353E-06,
            'my_t_lr': 0.000085183833089066,
            'lam': 200,
            'k': 1
        },
        'amazon_electronics_photo': {
            'num_layers': 6,
            'emb_dim': 64,
            'feat_drop': 0.8,
            'attn_drop': 0.2,
            'beta': 0,
            'lr': 0.001,
            'wd': 0.01,
            'my_lr': 3.5613262150520735e-06,
            'my_t_lr': 3.59169934474296e-05,
            'lam': 50,
            'k': 1
        },
    },
}


def set_configs(configs):
    configs = dict(
        configs, **predefined_configs[configs['teacher']][configs['dataset']])
    training_configs_path = Path.cwd().joinpath('train.conf.yaml')
    model_name = configs['student'] if configs['distill'] else configs['teacher']
    training_configs = get_training_config(training_configs_path, model_name)
    configs = dict(configs, **training_configs)
    configs['device'] = "/CPU:0"
    # configs['device'] = tlx.device('cpu')
    configs['division_seed'] = 0
    return configs


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())
    print()
    return itertools.starmap(Variant, itertools.product(*items.values()))


def func_search(trial):
    return {
        "my_lr": trial.suggest_uniform("my_lr", 1e-7, 1e-2),
        "my_t_lr": trial.suggest_uniform("my_t_lr", 1e-6, 1e-3),
    }


def main(args):
    # load_configs
    configs = set_configs(args.__dict__)  # 配置参数的字典格式
    # model_train
    variants = list(gen_variants(dataset=[configs['dataset']],
                                 model=[configs['model_name']],
                                 seed=[configs['seed']]))
    print(variants)
    results_dict = defaultdict(list)
    for variant in variants:
        results = raw_experiment(configs)
        print(results)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument('--teacher', type=str,
                        default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str,
                        default='GCN', help='Student Model')
    parser.add_argument('--distill', action='store_false',
                        default=True, help='Distill or not')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--labelrate', type=int, default=20,
                        help='label rate')  # 标签率，表示在训练数据中的标签比例。默认值是 20，表示仅有 20% 的数据有标签。
    parser.add_argument('--grad', type=int, default=1,
                        help='output grad or not')
    parser.add_argument('--automl', action='store_true',
                        default=False, help='Automl or not')  # 下面三个是自动机器学习参数
    parser.add_argument('--ntrials', type=int, default=8,
                        help='Number of trials')
    parser.add_argument('--njobs', type=int, default=1, help='Number of jobs')
    args = parser.parse_args()

    main(args)
