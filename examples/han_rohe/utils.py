import os
import errno
import datetime
import random
import numpy as np
import tensorlayerx as tlx
from sklearn.metrics import f1_score
from scipy import sparse, io as sio
from gammagl.utils import mask_to_index
from gammagl.data import HeteroGraph
import scipy.sparse as sp


def score(logits, labels):
    """
    Calculate accuracy, micro F1, and macro F1 scores.

    Parameters
    ----------
    logits : tensorlayerx.Tensor
        The predicted logits.
    labels : tensorlayerx.Tensor
        The true labels.

    Returns
    -------
    accuracy : float
        The accuracy score.
    micro_f1 : float
        The micro-averaged F1 score.
    macro_f1 : float
        The macro-averaged F1 score.
    """
    predictions = tlx.argmax(logits, axis=1)
    predictions = tlx.convert_to_numpy(predictions)
    labels = tlx.convert_to_numpy(labels)

    accuracy = np.sum(predictions == labels) / len(predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return accuracy, micro_f1, macro_f1


def score_detail(logits, labels):
    """
    Calculate detailed scores including accuracy per instance.

    Parameters
    ----------
    logits : tensorlayerx.Tensor
        The predicted logits.
    labels : tensorlayerx.Tensor
        The true labels.

    Returns
    -------
    accuracy : float
        The accuracy score.
    micro_f1 : float
        The micro-averaged F1 score.
    macro_f1 : float
        The macro-averaged F1 score.
    acc_detail : numpy.ndarray
        An array indicating correct (1) or incorrect (0) predictions per instance.
    """
    predictions = tlx.argmax(logits, axis=1)
    predictions = tlx.convert_to_numpy(predictions)
    labels = tlx.convert_to_numpy(labels)

    acc_detail = (predictions == labels).astype(int)
    accuracy = np.sum(acc_detail) / len(acc_detail)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    return accuracy, micro_f1, macro_f1, acc_detail


def evaluate(model, data, labels, mask, loss_func, detail=False):
    """
    Evaluate the model on the given data.

    Parameters
    ----------
    model : tensorlayerx.nn.Module
        The model to evaluate.
    data : dict
        A dictionary containing graph data.
import numpy as np
    labels : tensorlayerx.Tensor
        The true labels.
    mask : numpy.ndarray
        The indices of nodes to evaluate.
    loss_func : function
        The loss function.
    detail : bool
        Whether to return detailed accuracy per instance.

    Returns
    -------
    If detail is False:
        loss : float
            The loss value.
        accuracy : float
            The accuracy score.
        micro_f1 : float
            The micro-averaged F1 score.
        macro_f1 : float
            The macro-averaged F1 score.
    If detail is True:
        acc_detail : numpy.ndarray
            Accuracy per instance.
        accuracy : float
            The accuracy score.
        micro_f1 : float
            The micro-averaged F1 score.
        macro_f1 : float
            The macro-averaged F1 score.
    """
    model.set_eval()
    logits = model(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    logits = logits['paper']  # Adjust based on the node type
    mask_indices = mask  # Assuming mask is an array of indices
    logits_masked = tlx.gather(logits, tlx.convert_to_tensor(mask_indices, dtype=tlx.int64))
    labels_masked = tlx.gather(labels, tlx.convert_to_tensor(mask_indices, dtype=tlx.int64))
    loss = loss_func(logits_masked, labels_masked)
    if detail:
        accuracy, micro_f1, macro_f1, acc_detail = score_detail(logits_masked, labels_masked)
        return acc_detail, accuracy, micro_f1, macro_f1
    else:
        accuracy, micro_f1, macro_f1 = score(logits_masked, labels_masked)
        return loss, accuracy, micro_f1, macro_f1


def get_hg(dataname, given_adj_dict, features_dict):
    """
    Construct a HeteroGraph based on the dataset name and adjacency matrices.

    Parameters
    ----------
    dataname : str
        The name of the dataset.
    given_adj_dict : dict
        A dictionary of adjacency matrices.
    features_dict : dict
        A dictionary of node features for each node type.

    Returns
    -------
    graph : gammagl.data.HeteroGraph
        The constructed heterogeneous graph.
    """
    if dataname == 'acm':
        edge_index_dict = {
            ('paper', 'pa', 'author'): np.array(given_adj_dict['pa'].nonzero()),
            ('author', 'ap', 'paper'): np.array(given_adj_dict['ap'].nonzero()),
            ('paper', 'pf', 'field'): np.array(given_adj_dict['pf'].nonzero()),
            ('field', 'fp', 'paper'): np.array(given_adj_dict['fp'].nonzero()),
        }
    elif dataname == 'aminer':
        edge_index_dict = {
            ('paper', 'pa', 'author'): np.array(given_adj_dict['pa'].nonzero()),
            ('author', 'ap', 'paper'): np.array(given_adj_dict['ap'].nonzero()),
            ('paper', 'pr', 'ref'): np.array(given_adj_dict['pr'].nonzero()),
            ('ref', 'rp', 'paper'): np.array(given_adj_dict['rp'].nonzero()),
        }
    elif dataname == 'dblp':
        edge_index_dict = {
            ('paper', 'pa', 'author'): np.array(given_adj_dict['pa'].nonzero()),
            ('author', 'ap', 'paper'): np.array(given_adj_dict['ap'].nonzero()),
            ('paper', 'pc', 'conf'): np.array(given_adj_dict['pc'].nonzero()),
            ('conf', 'cp', 'paper'): np.array(given_adj_dict['cp'].nonzero()),
            ('paper', 'pt', 'term'): np.array(given_adj_dict['pt'].nonzero()),
            ('term', 'tp', 'paper'): np.array(given_adj_dict['tp'].nonzero()),
        }
    elif dataname == 'yelp':
        edge_index_dict = {
            ('b', 'bu', 'u'): np.array(given_adj_dict['bu'].nonzero()),
            ('u', 'ub', 'b'): np.array(given_adj_dict['ub'].nonzero()),
            ('b', 'bs', 's'): np.array(given_adj_dict['bs'].nonzero()),
            ('s', 'sb', 'b'): np.array(given_adj_dict['sb'].nonzero()),
            ('b', 'bl', 'l'): np.array(given_adj_dict['bl'].nonzero()),
            ('l', 'lb', 'b'): np.array(given_adj_dict['lb'].nonzero()),
        }
    else:
        raise ValueError(f"Dataset {dataname} is not supported.")

    num_nodes_dict = {ntype: features_dict[ntype].shape[0] for ntype in features_dict.keys()}
    graph = HeteroGraph(edge_index_dict=edge_index_dict, num_nodes_dict=num_nodes_dict, x_dict=features_dict)
    return graph


def set_random_seed(seed=0):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)


def mkdir_p(path, log=True):
    """
    Create a directory for the specified path.

    Parameters
    ----------
    path : str
        Path name.
    log : bool
        Whether to print result for directory creation.
    """
    try:
        os.makedirs(path)
        if log:
            print(f'Created directory {path}')
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print(f'Directory {path} already exists.')
        else:
            raise


def get_date_postfix():
    """
    Get a date-based postfix for directory name.

    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = f'{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}'
    return post_fix


def setup_log_dir(args, sampling=False):
    """
    Name and create directory for logging.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration arguments.
    sampling : bool
        Whether we are using sampling-based training.

    Returns
    -------
    log_dir : str
        Path for logging directory.
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args.log_dir,
        f'{args.dataset}_{date_postfix}'
    )

    if sampling:
        log_dir += '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 200,
    'patience': 100
}


def setup(args):
    """
    Set up the configuration and environment.

    Parameters
    ----------
    args : dict or argparse.Namespace
        Configuration arguments.

    Returns
    -------
    args : dict or argparse.Namespace
        Updated configuration arguments.
    """
    args.__dict__.update(default_configure)
    set_random_seed(args.seed)
    args.dataset = 'ACMRaw' if args.hetero else 'ACM'
    args.device = 'CPU'
    args.log_dir = setup_log_dir(args)
    # tlx.set_device(device_name=args.device)
    return args


def get_binary_mask(total_size, indices):
    """
    Create a binary mask given indices.

    Parameters
    ----------
    total_size : int
        The total size of the mask.
    indices : array-like
        The indices to be set to True.

    Returns
    -------
    mask : numpy.ndarray
        The binary mask array.
    """
    mask = np.zeros(total_size, dtype=bool)
    mask[indices] = True
    return mask


def load_acm_raw():
    data_path = 'acm/ACM.mat'
    data = sio.loadmat(data_path)
    p_vs_f = data['PvsL']  # paper-field adjacency
    p_vs_a = data['PvsA']  # paper-author adjacency
    p_vs_t = data['PvsT']  # paper-term feature matrix
    p_vs_c = data['PvsC']  # paper-conference labels

    # 我们将分配以下类别
    # (1) KDD papers -> class 0 (data mining)
    # (2) SIGMOD and VLDB papers -> class 1 (database)
    # (3) SIGCOMM and MOBICOMM papers -> class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    # 筛选我们关注的会议论文
    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = np.nonzero(p_vs_c_filter.sum(1))[0]  # 筛选有标签的论文
    p_vs_f = p_vs_f[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    # 构建边索引
    edge_index_pa = np.vstack(p_vs_a.nonzero())  # (2, num_edges) -> (source, target)
    edge_index_ap = edge_index_pa[[1, 0]]  # 反转 -> (target, source)
    edge_index_pf = np.vstack(p_vs_f.nonzero())
    edge_index_fp = edge_index_pf[[1, 0]]

    # 构建节点特征字典
    features = tlx.convert_to_tensor(p_vs_t.toarray(), dtype=tlx.float32)
    features_dict = {'paper': features}

    # 处理标签
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = tlx.convert_to_tensor(labels, dtype=tlx.int64)

    num_classes = 3

    # 创建 train, val, test 索引
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_p[pc_c_mask]] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = features.shape[0]
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx] = True
    test_mask = np.zeros(num_nodes, dtype=bool)
    test_mask[test_idx] = True

    # 构建原始 HeteroGraph
    graph = HeteroGraph()
    graph['paper'].x = features_dict['paper']
    graph['paper'].num_nodes = num_nodes
    graph['author'].num_nodes = p_vs_a.shape[1]
    graph['field'].num_nodes = p_vs_f.shape[1]

    # 添加边
    graph['paper', 'pa', 'author'].edge_index = edge_index_pa
    graph['author', 'ap', 'paper'].edge_index = edge_index_ap
    graph['paper', 'pf', 'field'].edge_index = edge_index_pf
    graph['field', 'fp', 'paper'].edge_index = edge_index_fp

    # 为 paper 节点添加标签和掩码
    graph['paper'].y = labels
    graph['paper'].train_mask = train_mask
    graph['paper'].val_mask = val_mask
    graph['paper'].test_mask = test_mask

    # 创建基于元路径的图
    # 元路径 PAP: 从 paper 到 author 再回到 paper
    pap_adj = p_vs_a.dot(p_vs_a.T)
    pap_edge_index = np.vstack(pap_adj.nonzero())  # 从 paper 到 paper

    # 元路径 PFP: 从 paper 到 field 再回到 paper
    pfp_adj = p_vs_f.dot(p_vs_f.T)
    pfp_edge_index = np.vstack(pfp_adj.nonzero())  # 从 paper 到 paper

    # 构建元路径图（PAP 和 PFP）
    meta_graph = HeteroGraph()
    meta_graph['paper'].x = features_dict['paper']
    meta_graph['paper'].num_nodes = num_nodes

    # 添加基于元路径的边
    meta_graph['paper', 'author', 'paper'].edge_index = pap_edge_index
    meta_graph['paper', 'field', 'paper'].edge_index = pfp_edge_index

    # 为 meta_graph 添加标签和掩码
    meta_graph['paper'].y = labels
    meta_graph['paper'].train_mask = train_mask
    meta_graph['paper'].val_mask = val_mask
    meta_graph['paper'].test_mask = test_mask

    return graph, meta_graph, features_dict, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask



def load_data(dataset):
    """
    Load data based on the dataset name.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    Returns
    -------
    Varies depending on the dataset.
    """
    if dataset == 'ACMRaw':
        return load_acm_raw()
    else:
        raise NotImplementedError(f'Unsupported dataset {dataset}')


class EarlyStopping(object):
    """
    Early stopping utility to stop training when a monitored metric has stopped improving.
    """

    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = f'early_stop_{dt.date()}_{dt.hour:02d}-{dt.minute:02d}-{dt.second:02d}.npz'
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        """
        Update the early stopping counter and check whether to stop training.

        Parameters
        ----------
        loss : float
            The current loss value.
        acc : float
            The current accuracy.
        model : tensorlayerx.nn.Module
            The model being trained.

        Returns
        -------
        early_stop : bool
            Whether training should be stopped.
        """
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = min(loss, self.best_loss)
            self.best_acc = max(acc, self.best_acc)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """
        Save the current model parameters.

        Parameters
        ----------
        model : tensorlayerx.nn.Module
            The model to save.
        """
        model.save_weights(self.filename, format='npz_dict')

    def load_checkpoint(self, model):
        """
        Load the model parameters from the saved checkpoint.

        Parameters
        ----------
        model : tensorlayerx.nn.Module
            The model to load parameters into.
        """
        model.load_weights(self.filename, format='npz_dict')
