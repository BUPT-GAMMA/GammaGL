import os

os.environ['TL_BACKEND'] = 'torch'
import argparse
from hypersearch import raw_experiment, AutoML

predefined_configs = {
    'GCN': {
        'cora': {
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 0.0010048137731444346,
            'my_t_lr': 0.00013337740190874308,
            'lam': 1,  # 正则化参数
            'k': 1
        },
        'citeseer': {
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 0.0008736877838717051,
            'my_t_lr': 0.0019445362780140703,
            'lam': 1,
            'k': 1
        },
        'pubmed': {
            'hidden_dim': 64,
            'drop_rate': 0.8,
            'num_layers': 2,
            'my_lr': 7.39520400332059E-07,
            'my_t_lr': 0.0000937937968682644,
            'lam': 1000,
            'k': 1
        },
    },
    'GAT': {
        'cora': {
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.0012033602084655297,
            'my_t_lr': 0.0009098411162144382,
            'lam': 1,
            'k': 2
        },
        'citeseer': {
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.000858867483786378,
            'my_t_lr': 0.00156837256987387,
            'lam': 1,
            'k': 2
        },
        'pubmed': {
            'hidden_dim': 8,
            'heads': 8,
            'drop_rate': 0.6,
            'num_layers': 2,
            'my_lr': 0.0000295072855477947,
            'my_t_lr': 0.000003722992284806,
            'lam': 200,
            'k': 2
        },
    },
}


def set_configs(configs):
    configs = dict(
        configs, **predefined_configs[configs['teacher']][configs['dataset']])
    return configs


def func_search(trial):
    return {
        "my_lr": trial.suggest_uniform("my_lr", 1e-8, 0.004),
        "my_t_lr": trial.suggest_uniform("my_t_lr", 1e-6, 0.01),
    }


def main(args):
    configs = set_configs(args.__dict__)
    if configs['automl']:
        tool = AutoML(kwargs=configs, func_search=func_search)
        best_acc_test, mylr, tlr = tool.run()
        configs['my_lr'] = mylr
        configs['my_t_lr'] = tlr
        raw_experiment(configs)
    else:
        best_acc_test = raw_experiment(configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pubmed', help='Dataset')
    parser.add_argument("--dataset_path", type=str, default=r'../', help="path to save dataset")
    parser.add_argument('--teacher', type=str,
                        default='GCN', help='Teacher Model')
    parser.add_argument('--student', type=str,
                        default='GCN', help='Student Model')
    parser.add_argument('--automl', action='store_true',
                        default=True, help='Automl or not')
    parser.add_argument('--ntrials', type=int, default=30,
                        help='Number of trials')
    parser.add_argument('--njobs', type=int, default=1, help='Number of jobs')
    parser.add_argument("--max_epoch", type=int, default=300, help="max number of epoch")
    parser.add_argument("--patience", type=int, default=50, help="early stopping epoch")
    args = parser.parse_args()

    main(args)
