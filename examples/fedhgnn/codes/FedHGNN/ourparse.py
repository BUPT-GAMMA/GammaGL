import argparse
import os
os.environ['TL_BACKEND'] = 'torch'
import torch
def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--fea_dim', type=int, default=64, help='Dim of feature vectors.')
    parser.add_argument('--in_dim', type=int, default=64, help='Dim of input vectors.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dim of hidden vectors.')
    parser.add_argument('--out_dim', type=int, default=64, help='Dim of output vectors.')
    parser.add_argument('--shared_num', type=int, default=20, help='Dim of output vectors.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='acm', help='Choose a dataset.')#lastfm
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else 'cpu',
                        help='Which device to run the model.')
    parser.add_argument('--num_heads', type=list, default=[2],help='attention_head')
    parser.add_argument('--eps', type=float, default=1, help='total privacy budget.')
    parser.add_argument('--num_sample', type=int, default=0, help='number of sampled neighbors.')
    parser.add_argument('--valid_step', type=int, default=5, help='valid step.')
    parser.add_argument('--nonlinearity', type=str, default="relu", help='Which device to run the model.')
    parser.add_argument('--log_dir', type=str, default="../../log/", help='Which device to run the model.')
    parser.add_argument('--is_gcn', type=bool, default=False, help="whether using gcn")
    parser.add_argument('--is_attention', type=bool, default=False, help="whether using attention")
    parser.add_argument('--hetero', type=bool, default=True, help="whether using attention")
    parser.add_argument('--is_trans', type=bool, default=False, help="whether using trans")
    parser.add_argument('--is_random_init', type=bool, default=True, help="whether random user and item")
    parser.add_argument('--is_graph', type=bool, default=True, help="whether using graph")
    parser.add_argument('--local_train_num', type=int, default=1, help='Dim of latent vectors.')
    parser.add_argument('--agg_mode', type=str, default="add", help='Dim of latent vectors.')
    parser.add_argument('--agg_func', type=str, default="ATTENTION", help='Dim of latent vectors.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='lr weight_decay in optimizer.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--l2_reg', type=bool, default=True, help='L2 norm regularization in loss.')
    parser.add_argument('--grad_limit', type=float, default=1.0, help='Limit of l2-norm of item gradients.')
    parser.add_argument('--clients_limit', type=float, default=0.1, help='Limit of proportion of malicious clients.')
    parser.add_argument('--items_limit', type=int, default=60, help='Limit of number of non-zero item gradients.')
    parser.add_argument('--type', type=str, default="ATTENTION", help='Dim of latent vectors.')
    parser.add_argument('--p1', type=float, default=1, help='lr weight_decay in optimizer.')
    parser.add_argument('--p2', type=float, default=1, help='lr weight_decay in optimizer.')
    parser.add_argument("-f", "--file", default="file")
    return parser.parse_args()


args = parse_args()