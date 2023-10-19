import os
import tensorlayerx as tlx
import torch
from tensorlayerx.model import TrainOneStep, WithLoss
import numpy as np
import sys

sys.path.insert(0, os.path.abspath('../../'))  # adds path2gammagl to execute in command line.
from gammagl.utils import add_self_loops, remove_self_loops
from gammagl.mpops import *
from gammagl.models import GCNModel, GATModel, GraphSAGE_Sample_Model, APPNPModel, MLP, SGCModel


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()

    return result / len(labels)


def choose_model(configs, dataset):
    if configs['model_name'] == 'GCN':
        model = GCNModel(feature_dim=dataset.num_node_features,
                         hidden_dim=configs['hidden'],
                         num_class=dataset.num_classes,
                         drop_rate=configs['dropout'],
                         num_layers=1,
                         norm='both',
                         name='GCN')
    elif configs['model_name'] == 'GAT':
        model = GATModel(feature_dim=dataset.num_node_features,
                         hidden_dim=8,
                         num_class=dataset.num_classes,
                         heads=8,
                         drop_rate=0.6,
                         num_layers=1)
    elif configs['model_name'] == 'GraphSAGE':
        model = GraphSAGE_Sample_Model(in_feat=dataset.num_node_features,
                                       hid_feat=16,
                                       out_feat=dataset.num_classes,
                                       drop_rate=0.5,
                                       num_layers=1)
    elif configs['model_name'] == 'APPNP':
        model = APPNPModel(feature_dim=dataset.num_node_features,
                           num_class=dataset.num_classes,
                           iter_K=10,
                           alpha=0.1,
                           drop_rate=0.5)
    elif configs['model_name'] == 'MLP':
        model = MLP(in_channels=dataset.num_node_features,
                    hidden_channels=configs['hidden'],
                    out_channels=dataset.num_classes,
                    num_layers=2,
                    dropout=configs['dropout'])
    elif configs['model_name'] == 'SGC':
        model = SGCModel(feature_dim=dataset.num_node_features,
                         num_class=dataset.num_classes,
                         iter_K=2)
    else:
        raise ValueError(f'Undefined Model.')
    return model


def distill_loss(configs, labels_one_hot, student_hard, teacher_softmax, train_mask):
    hard_loss = -tlx.ops.reduce_sum(
        (labels_one_hot[train_mask] + 1e-6) * tlx.ops.log(student_hard[train_mask] + 1e-6))
    soft_loss = -tlx.ops.reduce_sum((teacher_softmax + 1e-6) * tlx.ops.log(student_hard + 1e-6))
    loss = soft_loss + configs['lam'] * hard_loss
    return loss


def evaluate_loss(logits, y, num_classes, val_mask):
    student_hard = tlx.ops.softmax(logits, axis=1)
    student_logits = student_hard[val_mask]
    labels_one_hot = tlx.nn.OneHot(depth=num_classes)(y)[val_mask]
    loss = -tlx.ops.reduce_sum(
        (labels_one_hot + 1e-6) * tlx.ops.log(student_logits + 1e-6))
    return loss


class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, graph, configs, nei_entropy, t_model, teacher_logits, num_classes, train_mask):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.configs = configs
        self.graph = graph
        self.nei_entropy = nei_entropy
        self.t_model = t_model
        self.teacher_logits = teacher_logits
        self.num_classes = num_classes
        self.train_mask = train_mask

    def forward(self, data, y):
        edge_index, _ = add_self_loops(self.graph.edge_index, num_nodes=self.graph.num_nodes)
        if self.configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
            logits = self.backbone_network(data, edge_index, None, self.graph.num_nodes)
        if self.configs['model_name'] in ['APPNP']:
            logits = self.backbone_network(data)
        elif self.configs['model_name'] == 'GAT':
            logits, _, fs = self.backbone_network(data)
        elif self.configs['model_name'] in ['GraphSAGE', 'SGC']:
            logits = self.backbone_network(data)
        with torch.no_grad():
            f_logits = torch.norm(logits, 2, dim=1)  # L2 范数
        extract_x = tlx.ops.concat(
            [logits, tlx.expand_dims(f_logits, axis=1), tlx.expand_dims(self.nei_entropy, axis=1)],
            axis=1)

        temparature = self.t_model(extract_x)
        temparature = (temparature - 0.2) * self.configs['k']
        temparature = tlx.ops.where(tlx.ops.abs(temparature) < 0.0001, 0.001 * tlx.ops.ones_like(temparature),
                                    temparature)

        teacher_logits_t = tlx.ops.transpose(
            tlx.ops.divide(tlx.ops.transpose(self.teacher_logits), tlx.ops.squeeze(temparature, axis=1)))
        teacher_softmax = tlx.ops.softmax(teacher_logits_t, axis=1)
        student_hard = tlx.ops.softmax(logits, axis=1)
        labels_one_hot = tlx.nn.OneHot(depth=self.num_classes)(y)
        loss = self._loss_fn(self.configs, labels_one_hot, student_hard, teacher_softmax, self.train_mask)
        return loss


class t_SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn, graph, configs, num_classes, model, val_mask):
        super(t_SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)
        self.graph = graph
        self.configs = configs
        self.num_classes = num_classes
        self.model = model
        self.val_mask = val_mask

    def forward(self, data, y):
        edge_index, _ = add_self_loops(self.graph.edge_index, num_nodes=self.graph.num_nodes)
        if self.configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
            logits = self.model(data, edge_index, None, self.graph.num_nodes)
        if self.configs['model_name'] in ['APPNP']:
            logits = self.model(data)
        elif self.configs['model_name'] == 'GAT':
            logits, _, fs = self.model(data)
        elif self.configs['model_name'] in ['GraphSAGE', 'SGC']:
            logits = self.model(data)
        loss = self._loss_fn(logits, y, self.num_classes, self.val_mask)
        return loss


def zero_grad(my_model):  # grad->0
    with torch.no_grad():
        for p in my_model.parameters():
            if p.grad is not None:
                p.grad.zero_()


def distill_train(dur, epoch, model, configs, graph, dataset, temp, nei_entropy, t_model, teacher_logits, train_mask,
                  val_mask, my_val_mask):
    model.set_train()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    with torch.no_grad():
        f_logits = torch.norm(logits, 2, dim=1)  # L2 范数
    extract_x = tlx.ops.concat(
        [logits, tlx.expand_dims(f_logits, axis=1), tlx.expand_dims(nei_entropy, axis=1)],
        axis=1)

    temparature = t_model(extract_x)
    temparature = (temparature - 0.2) * configs['k']
    temparature = tlx.ops.where(tlx.ops.abs(temparature) < 0.0001, 0.001 * tlx.ops.ones_like(temparature),
                                temparature)

    teacher_logits_t = tlx.ops.transpose(
        tlx.ops.divide(tlx.ops.transpose(teacher_logits), tlx.ops.squeeze(temparature, axis=1)))
    teacher_softmax = tlx.ops.softmax(teacher_logits_t, axis=1)
    student_hard = tlx.ops.softmax(logits, axis=1)
    labels_one_hot = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y)
    hard_loss = -tlx.ops.reduce_sum(
        (labels_one_hot[train_mask] + 1e-6) * tlx.ops.log(student_hard[train_mask] + 1e-6))
    soft_loss = -tlx.ops.reduce_sum((teacher_softmax + 1e-6) * tlx.ops.log(student_hard + 1e-6))
    train_loss = soft_loss + configs['lam'] * hard_loss
    model_dict = {}
    for p_name, p in model.named_parameters():
        agr = torch.autograd.grad(train_loss, p, create_graph=True)[0]
        model_dict[p_name] = p - configs['my_lr'] * agr  # update model 【公式10】
    model.load_state_dict(model_dict)

    t_train_loss = torch.tensor(0)
    t_model_dict = {}
    if epoch > 20:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
            logits = model(graph.x, edge_index, None, graph.num_nodes)
        student_hard = tlx.ops.softmax(logits, axis=1)
        student_logits = student_hard[val_mask]
        labels_one_hot = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y)[val_mask]
        t_train_loss = -tlx.ops.reduce_sum(
            (labels_one_hot + 1e-6) * tlx.ops.log(student_logits + 1e-6))

        for pvt_name, pt in t_model.named_parameters():  # t1,t2
            pt.grad = torch.autograd.grad(t_train_loss, pt, create_graph=True)[0]  # 求model中的参数对tmodel中参数的梯度
            t_model_dict[pvt_name] = pt - configs['my_t_lr'] * pt.grad

        t_model.load_state_dict(t_model_dict)
        zero_grad(t_model)
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    if configs['model_name'] in ['APPNP']:
        logits = model(graph.x)
    elif configs['model_name'] == 'GAT':
        logits, _, fs = model(graph.x)
    elif configs['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(graph.x)
    logp = tlx.ops.log(tlx.ops.softmax(logits))
    acc_val = accuracy(logp[my_val_mask], graph.y[my_val_mask])
    loss_val = 0
    print('Epoch %d | acc_val: %.4f | distill_loss: %.4f | evaluate_loss: %.4f' % (
    epoch, acc_val.item(), train_loss.item(), t_train_loss.item()))
    return acc_val, loss_val


def split_train_mask(graph, num_classes):
    label = graph.y[graph.train_mask].numpy()
    indices = np.where(graph.train_mask.numpy())[0]
    class_indices = [indices[np.where(label == c)[0]] for c in range(num_classes)]
    np_train_mask = np.zeros(graph.num_nodes)
    np_val_mask = np.zeros(graph.num_nodes)
    np_my_val_mask = np.zeros(graph.num_nodes)
    for i in range(num_classes):
        np.random.shuffle(class_indices[i])
        np_train_mask[class_indices[i][:20]] = 1
        np_val_mask[class_indices[i][20:40]] = 1
        np_my_val_mask[class_indices[i][40:]] = 1
    train_mask = tlx.ops.convert_to_tensor(np_train_mask).bool()
    val_mask = tlx.ops.convert_to_tensor(np_val_mask).bool()
    my_val_mask = tlx.ops.convert_to_tensor(np_my_val_mask).bool()
    return train_mask, val_mask, my_val_mask


def model_train(configs, model, graph, teacher_logits, dataset):
    all_logits = []
    dur = []
    best = 0
    cnt = 0
    epoch = 1

    teacher_softmax = tlx.ops.softmax(teacher_logits, axis=1)
    edge_index_no_self_loops, _ = remove_self_loops(graph.edge_index)
    msg = tlx.gather(teacher_softmax, edge_index_no_self_loops[1])
    nei_logits_sum = unsorted_segment_sum(msg, edge_index_no_self_loops[0], graph.num_nodes)
    nei_num = unsorted_segment_sum(tlx.ops.ones_like(graph.edge_index[1, :]), edge_index_no_self_loops[0],
                                   graph.num_nodes)  # 分段统计
    nei_probability = tlx.ops.transpose(tlx.ops.divide(tlx.ops.transpose(nei_logits_sum), nei_num))  # 邻居汇总/邻居个数
    nei_entropy = -tlx.ops.reduce_sum(nei_probability * tlx.ops.log(nei_probability), axis=1)
    for i in range(len(nei_entropy)):
        if nei_entropy[i] != nei_entropy[i]:
            nei_entropy[i] = 0.0001

    t_model = MLP(num_layers=2, in_channels=dataset.num_classes + 2, hidden_channels=64, out_channels=1, dropout=0.6)

    temp = 0
    best = 0
    train_mask, val_mask, my_val_mask = split_train_mask(graph, dataset.num_classes)
    print("train_val_my %d %d %d" % (
    torch.sum(train_mask).item(), torch.sum(val_mask).item(), torch.sum(my_val_mask).item()))
    while epoch < configs['max_epoch']:
        acc_val, loss_val = distill_train(dur, epoch, model, configs, graph, dataset, temp, nei_entropy, t_model,
                                          teacher_logits, train_mask, val_mask, my_val_mask)
        epoch += 1
        if epoch > 0:
            if acc_val >= best:
                best = acc_val
                # 保存模型状态
                model.save_weights(model.name + ".npz", format='npz_dict')
                best_epoch = epoch
                cnt = 0
            else:
                cnt += 1
            if cnt == configs['patience'] or epoch == configs['max_epoch'] or loss_val != loss_val:
                print("Stop!!!")
                print('best_epoch: %d' % best_epoch)
                break
    # 载入模型状态
    model.load_weights(model.name + ".npz", format='npz_dict')
    print("Optimization Finished!")
    distill_test(model, graph, configs, teacher_logits)
    return best


def distill_test(model, graph, configs, teacher_logits):
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    if configs['model_name'] in ['APPNP']:
        logits = model(graph.x)
    elif configs['model_name'] == 'GAT':
        logits, _, fs = model(graph.x)
    elif configs['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(graph.x)
    loss_test = tlx.losses.softmax_cross_entropy_with_logits(logits, graph.y)
    logp = tlx.ops.log(tlx.ops.softmax(logits))
    acc_test = accuracy(logp[graph.test_mask], graph.y[graph.test_mask])
    acc_teacher_test = accuracy(teacher_logits[graph.test_mask], graph.y[graph.test_mask])
    preds = tlx.ops.argmax(logp, axis=1)
    teacher_preds = tlx.ops.argmax(teacher_logits, axis=1)
    same_predict = np.count_nonzero(
        teacher_preds[graph.test_mask] == preds[graph.test_mask]) / 2358
    acc_dis = np.abs(acc_teacher_test.item() - acc_test.item())
    print(
        "Test set results: loss= {:.4f} acc_test= {:.4f} acc_teacher_test= {:.4f} acc_dis={:.4f} same_predict= {:.4f}".format(
            loss_test.item(), acc_test.item(), acc_teacher_test.item(), acc_dis, same_predict))
