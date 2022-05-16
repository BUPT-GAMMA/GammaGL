import argparse



def main(args):
    TODO()




















if __name == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others);' + 
                         '5 - only term features (zero vec for others).')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    parser.add_argument('--heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    parser.add_argument('--n_epoch', type=int, default=300, help='Number of epochs.')
    parser.add_argument('--patience', type=int, default=30, help='Patience.')
    parser.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--slope', type=float, default=0.05)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--edge_feats', type=int, default=64)
    parser.add_argument('--run', type=int, default=1)

    args = parser.parse_args()
    main()