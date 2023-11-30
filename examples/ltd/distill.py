import os
import tensorlayerx as tlx
import numpy as np
import sys

import torch.nn

sys.path.insert(0, os.path.abspath('../../'))
from gammagl.utils import add_self_loops, remove_self_loops
from gammagl.mpops import *
from gammagl.models import GCNModel, GATModel, MLP


def layer_normalize(logits):
    # 计算每个样本的均值和标准差
    mean = torch.mean(logits, dim=-1, keepdim=True)
    std = torch.std(logits, dim=-1, keepdim=True)

    # 层归一化
    normalized_logits = (logits - mean) / (std + 1e-8)

    return normalized_logits

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    result = correct.sum()

    return result / len(labels)

def accuracy_t(output, labels):
    preds = output.max(1)[1].type_as(labels)
    label = labels.max(1)[1]
    correct = preds.eq(label).double()
    result = correct.sum()

    return result / len(labels)


def choose_model(configs, dataset):
    if configs['student'] == 'GCN':
        model = GCNModel(feature_dim=dataset.num_node_features,
                         hidden_dim=configs['hidden_dim'],
                         num_class=dataset.num_classes,
                         drop_rate=configs['drop_rate'],
                         num_layers=configs['num_layers'],
                         norm='both',
                         name='GCN')
    elif configs['student'] == 'GAT':
        model = GATModel(feature_dim=dataset.num_node_features,
                         hidden_dim=configs['hidden_dim'],
                         num_class=dataset.num_classes,
                         heads=configs['heads'],
                         drop_rate=configs['drop_rate'],
                         num_layers=configs['num_layers'],
                         name="GAT",
                         )
    else:
        raise ValueError(f'Undefined Model.')
    return model


def zero_grad(my_model):
    with torch.no_grad():
        for p in my_model.parameters():
            if p.grad is not None:
                p.grad.zero_()


def distill_train(epoch, model, configs, graph, nei_entropy, t_model, teacher_logits, train_mask,
                  val_mask, my_val_mask, dataset):
    model.set_train()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['student'] == 'GCN':
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    elif configs['student'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    # if configs['student'] == 'GAT' and configs['dataset'] == 'citeseer':
    logits = layer_normalize(logits)
    acc_train = accuracy(logits[train_mask], graph.y[train_mask])
    acc_same = accuracy_t(logits, teacher_logits)
    with torch.no_grad():
        f_logits = torch.norm(logits, 2, dim=1)
    extract_x = tlx.ops.concat(
        [logits, tlx.expand_dims(f_logits, axis=1), tlx.expand_dims(nei_entropy, axis=1)],
        axis=1)

    temparature = t_model(extract_x)
    temparature = (temparature + 0.2) * configs['k']
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
        model_dict[p_name] = p - configs['my_lr'] * agr
    model.load_state_dict(model_dict)

    t_train_loss = torch.tensor(0)
    if epoch > 20:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        if configs['student'] == 'GCN':
            logits = model(graph.x, edge_index, None, graph.num_nodes)
        elif configs['student'] == 'GAT':
            logits = model(graph.x, edge_index, graph.num_nodes)
        # if configs['student'] == 'GAT' and configs['dataset'] == 'citeseer':
        logits = layer_normalize(logits)
        student_hard = tlx.ops.softmax(logits, axis=1)
        labels_one_hot = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y)
        t_train_loss = -tlx.ops.reduce_sum(
            (labels_one_hot[val_mask] + 1e-6) * tlx.ops.log(student_hard[val_mask] + 1e-6))

        for name, t in model.named_parameters():
            t_grad = torch.autograd.grad(t_train_loss, t, create_graph=True, allow_unused=True)[0]
            t_grad = t_grad.detach()
            for p_name, p in model.named_parameters():
                if t_grad.shape == p.shape:
                    model_dict[p_name] = model_dict[p_name] * t_grad
        t_model_dict = {}
        for p_name, p in model.named_parameters():
            for name, t in t_model.named_parameters():
                agr = torch.autograd.grad(tlx.reduce_sum(model_dict[p_name]), t, create_graph=True, allow_unused=True)[
                    0]
                t_model_dict[name] = t - configs['my_t_lr'] * agr

        t_model.load_state_dict(t_model_dict)
        zero_grad(t_model)
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['student'] == 'GCN':
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    elif configs['student'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    acc_val = accuracy(logits[my_val_mask], graph.y[my_val_mask])
    print(
        'Epoch %d | acc_train: %.4f| acc_val: %.4f | acc_same: %.4f | hard_loss: %.4f | soft_loss: %.4f| distill_loss: %.4f | evaluate_loss: %.4f' % (
            epoch, acc_train.item(), acc_val.item(), acc_same.item(), hard_loss.item(), soft_loss.item(),
            train_loss.item(),
            t_train_loss.item()))
    return acc_val, train_loss


def split_train_mask(graph, num_classes, train_per_class, val_per_class, my_val_per_class):
    label = graph.y[graph.train_mask].numpy()
    indices = np.where(graph.train_mask.numpy())[0]
    class_indices = [indices[np.where(label == c)[0]] for c in range(num_classes)]
    np_train_mask = np.zeros(graph.num_nodes)
    np_val_mask = np.zeros(graph.num_nodes)
    np_my_val_mask = np.zeros(graph.num_nodes)
    for i in range(num_classes):
        np.random.shuffle(class_indices[i])
        np_train_mask[class_indices[i][:train_per_class]] = 1
        np_val_mask[class_indices[i][train_per_class:train_per_class + val_per_class]] = 1
        np_my_val_mask[
            class_indices[i][train_per_class + val_per_class:train_per_class + val_per_class + my_val_per_class]] = 1
    train_mask = tlx.ops.convert_to_tensor(np_train_mask).bool()
    val_mask = tlx.ops.convert_to_tensor(np_val_mask).bool()
    my_val_mask = tlx.ops.convert_to_tensor(np_my_val_mask).bool()
    return train_mask, val_mask, my_val_mask


def model_train(configs, model, graph, teacher_logits, dataset):
    teacher_softmax = tlx.ops.softmax(teacher_logits, axis=1)
    edge_index_no_self_loops, _ = remove_self_loops(graph.edge_index)
    msg = tlx.gather(teacher_softmax, edge_index_no_self_loops[1])
    nei_logits_sum = unsorted_segment_sum(msg, edge_index_no_self_loops[0], graph.num_nodes)
    nei_num = unsorted_segment_sum(tlx.ops.ones_like(edge_index_no_self_loops[0]), edge_index_no_self_loops[0],
                                   graph.num_nodes)
    nei_probability = tlx.ops.transpose(tlx.ops.divide(tlx.ops.transpose(nei_logits_sum), nei_num))
    nei_entropy = -tlx.ops.reduce_sum(nei_probability * tlx.ops.log(nei_probability), axis=1)
    for i in range(len(nei_entropy)):
        if nei_entropy[i] != nei_entropy[i]:
            nei_entropy[i] = 0.0001

    t_model = MLP(num_layers=2, in_channels=dataset.num_classes + 2, hidden_channels=64, out_channels=1, dropout=0.4,
                  norm=None)

    best = 0
    cnt = 0
    epoch = 1
    train_mask, val_mask, my_val_mask = split_train_mask(graph, dataset.num_classes, 20, 20, 10)
    while epoch < configs['max_epoch']:
        acc_val, train_loss = distill_train(epoch, model, configs, graph, nei_entropy, t_model,
                                teacher_logits, train_mask, val_mask, my_val_mask, dataset)
        if epoch > 0:
            if acc_val >= best:
                best = acc_val
                model.save_weights(model.name + ".npz", format='npz_dict')
                best_epoch = epoch
                cnt = 0
            else:
                cnt = cnt + 1
            if cnt == configs['patience'] or epoch == configs['max_epoch'] or train_loss != train_loss:
                print("Stop!!!")
                print('best_epoch: %d' % best_epoch)
                break
        epoch += 1
    model.load_weights(model.name + ".npz", format='npz_dict')
    print("Optimization Finished!")
    acc_test = distill_test(model, graph, configs, teacher_logits)
    return acc_test


def distill_test(model, graph, configs, teacher_logits):
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['student'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    if configs['student'] in ['APPNP']:
        logits = model(graph.x)
    elif configs['student'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    elif configs['student'] in ['GraphSAGE', 'SGC']:
        logits = model(graph.x)
    loss_test = tlx.losses.softmax_cross_entropy_with_logits(logits, graph.y)
    logp = tlx.ops.log(tlx.ops.softmax(logits))
    acc_test = accuracy(logp[graph.test_mask], graph.y[graph.test_mask])
    acc_teacher_test = accuracy(teacher_logits[graph.test_mask], graph.y[graph.test_mask])
    acc_dis = np.abs(acc_teacher_test.item() - acc_test.item())
    print(
        "Test set results: loss= {:.4f} acc_test= {:.4f} acc_teacher_test= {:.4f} acc_dis={:.4f}".format(
            loss_test.item(), acc_test.item(), acc_teacher_test.item(), acc_dis))
    return acc_test
