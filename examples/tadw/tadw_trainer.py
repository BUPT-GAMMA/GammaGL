import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TL_BACKEND'] = 'torch'
import argparse
import tensorlayerx as tlx
from gammagl.datasets import Planetoid
from gammagl.models import TADWModel
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

if tlx.BACKEND == 'torch':  # when the backend is torch and you want to use GPU
    try:
        tlx.set_device(device='GPU', id=6)
    except:
        print("GPU is not available")


def calculate_acc(train_z, train_y, test_z, test_y):
    clf = svm.LinearSVC(C=5.0)
    clf.fit(train_z, train_y)
    predict_y = clf.predict(test_z)
    return metrics.accuracy_score(test_y, predict_y)


def main(args):
    # load datasets
    if str.lower(args.dataset) not in ['cora', 'citeseer']:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    dataset = Planetoid(args.dataset_path, args.dataset)
    graph = dataset[0]
    edge_index = graph.edge_index
    model = TADWModel(edge_index=edge_index,
                      embedding_dim=args.embedding_dim,
                      lr=args.lr,
                      lamda=args.lamda,
                      svdft=args.svdft,
                      node_feature=graph.x,
                      name="TADW")
    data = {
        "x": graph.x,
        "y": graph.y,
        "edge_index": graph.edge_index,
        "num_nodes": graph.num_nodes,
    }
    best_test_acc = 0
    z_test = 0
    for epoch in range(args.n_epoch):
        model.set_train()
        train_loss = model.fit()
        model.set_eval()
        z = model.campute()
        train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(data['y']),
                                                                            test_size=0.5, shuffle=True)
        test_acc = calculate_acc(train_x, train_y, test_x, test_y)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            z_test = z
        print("Epoch [{:0>3d}] ".format(epoch + 1) \
              + "  train loss: {:.4f}".format(train_loss.item()) \
              + "  test acc: {:.4f}".format(test_acc))

    z = z_test
    train_x, test_x, train_y, test_y = model_selection.train_test_split(z, tlx.convert_to_numpy(graph.y),
                                                                        test_size=0.5, shuffle=True)
    test_acc = calculate_acc(train_x, train_y, test_x, test_y)
    print("Test acc:  {:.4f}".format(test_acc))
    return test_acc


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', help='dataset')
    parser.add_argument("--dataset_path", type=str, default=r'', help="path to save dataset")
    parser.add_argument("--best_model_path", type=str, default=r'./', help="path to save best model")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--n_epoch", type=int, default=50, help="number of epoch")
    parser.add_argument("--embedding_dim", type=int, default=500)
    parser.add_argument("--lamda", type=float, default=0.5)
    parser.add_argument("--svdft", type=int, default=300)

    args = parser.parse_args()

    main(args)
