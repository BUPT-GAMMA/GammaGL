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
import torch.optim as optim


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
    if configs['model_name'] == 'GCN':
        model = GCNModel(feature_dim=dataset.num_node_features,
                         hidden_dim=64,
                         num_class=dataset.num_classes,
                         drop_rate=0.8,
                         num_layers=2,
                         norm='both',
                         name='GCN')
    elif configs['model_name'] == 'GAT':
        model = GATModel(feature_dim=dataset.num_node_features,
                       hidden_dim=8,
                       num_class=dataset.num_classes,
                       heads=8,
                       drop_rate=0.4,
                       num_layers=3,
                       name="GAT",
                       )
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


def zero_grad(my_model):  # grad->0
    with torch.no_grad():
        for p in my_model.parameters():
            if p.grad is not None:
                p.grad.zero_()


def distill_train(epoch, model, configs, graph, nei_entropy, t_model, teacher_logits, train_mask,
                  val_mask, my_val_mask, dataset):
    model.set_train()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    elif configs['model_name'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    acc_train = accuracy(logits[train_mask], graph.y[train_mask])
    acc_same = accuracy_t(logits, teacher_logits)
    # acc_teacher = accuracy(teacher_logits, graph.y)
    with torch.no_grad():
        f_logits = torch.norm(logits, 2, dim=1)  # L2 范数
    extract_x = tlx.ops.concat(
        [logits, tlx.expand_dims(f_logits, axis=1), tlx.expand_dims(nei_entropy, axis=1)],
        axis=1)

    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # su = tlx.reduce_sum(nei_entropy)
    temparature = t_model(extract_x)
    t_max = torch.max(temparature)
    t_min = torch.min(temparature)
    temparature = (temparature + 0.2) * configs['k']
    temparature = tlx.ops.where(tlx.ops.abs(temparature) < 0.0001, 0.001 * tlx.ops.ones_like(temparature),
                                temparature)
    t_sum = tlx.reduce_sum(temparature)
    # t = tlx.ops.squeeze(temparature, axis=1)
    teacher_logits_t = tlx.ops.transpose(
        tlx.ops.divide(tlx.ops.transpose(teacher_logits), tlx.ops.squeeze(temparature, axis=1)))
    teacher_softmax = tlx.ops.softmax(teacher_logits_t, axis=1)
    # teacher_softmax = tlx.ops.softmax(teacher_logits, axis=1)
    student_hard = tlx.ops.softmax(logits, axis=1)
    labels_one_hot = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y)
    hard_loss = -tlx.ops.reduce_sum(
        (labels_one_hot[train_mask] + 1e-6) * tlx.ops.log(student_hard[train_mask] + 1e-6))
    # hard_loss = tlx.losses.softmax_cross_entropy_with_logits(logits[train_mask], graph.y[train_mask])
    soft_loss = -tlx.ops.reduce_sum((teacher_softmax + 1e-6) * tlx.ops.log(student_hard + 1e-6))
    # for p_name, p in model.named_parameters():
    #     for name, t in t_model.named_parameters():
    #         agr = torch.autograd.grad(tlx.reduce_sum(t), p, create_graph=True)[0]
    #         print(agr)
    train_loss = soft_loss + configs['lam'] * hard_loss
    # train_loss = 1 * soft_loss + 0 * hard_loss
    model_dict = {}

    grad_temp = {}
    for p_name, p in model.named_parameters():
        agr = torch.autograd.grad(train_loss, p, create_graph=True)[0]
        model_dict[p_name] = p - configs['my_lr'] * agr  # update model 【公式10】
        # for name, t in t_model.named_parameters():
        #     if t.requires_grad:
        #         g = torch.autograd.grad(tlx.reduce_sum(model_dict[p_name]), t, create_graph=True)[0]
        #         grad_temp[p_name] = {}
        #         grad_temp[p_name][name] = g
        # if epoch >= 20:
        #     for name, t in t_model.named_parameters():
        #         if t.requires_grad:
        #             g = torch.autograd.grad(tlx.reduce_sum(model_dict[p_name]), t, create_graph=True)[0]
        #             grad_temp[p_name] = {}
        #             grad_temp[p_name][name] = g

    model.load_state_dict(model_dict, strict=False)
    # zero_grad(model)

    # optimizer.zero_grad()
    # train_loss.backward()
    # optimizer.step()

    t_train_loss = torch.tensor(0)
    if epoch > 20:
        edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
        if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
            logits = model(graph.x, edge_index, None, graph.num_nodes)
        elif configs['model_name'] == 'GAT':
            logits = model(graph.x, edge_index, graph.num_nodes)
        student_hard = tlx.ops.softmax(logits, axis=1)
        student_logits = student_hard[val_mask]
        labels_one_hot = tlx.nn.OneHot(depth=dataset.num_classes)(graph.y)[val_mask]
        t_train_loss = -tlx.ops.reduce_sum(
            (labels_one_hot + 1e-6) * tlx.ops.log(student_logits + 1e-6))
        # t_train_loss = tlx.losses.softmax_cross_entropy_with_logits(logits[val_mask], graph.y[val_mask])

        # for name, t in model.named_parameters():
        #     print(name)
        #     print(t)
        # for pvt_name, pt in t_model.named_parameters():  # t1,t2
        #     if pt.requires_grad:
        #         pt.grad = torch.autograd.grad(train_loss, pt, create_graph=True)[0]  # 求model中的参数对tmodel中参数的梯度
        #         t_model_dict[pvt_name] = pt - configs['my_t_lr'] * pt.grad
        for name, t in model.named_parameters():
            t_grad = torch.autograd.grad(t_train_loss, t, create_graph=True, allow_unused=True)[0]
            t_grad = t_grad.detach()
            for p_name, p in model.named_parameters():
                if t_grad.shape == p.shape:
                    model_dict[p_name] = model_dict[p_name] * t_grad
        t_model_dict = {}
        for p_name, p in model.named_parameters():
            for name, t in t_model.named_parameters():
                # agr = torch.autograd.grad(model_dict[p_name], t, create_graph=True, allow_unused=True)[0]
                agr = torch.autograd.grad(tlx.reduce_sum(model_dict[p_name]), t, create_graph=True, allow_unused=True)[
                    0]
                t_model_dict[name] = t - configs['my_lr'] * agr

        t_model.load_state_dict(t_model_dict, strict=False)
        # for p_name, p_t in t_model.named_parameters():
        #     # if p_t.requires_grad and grad_temp[name][p_name]:
        #     # pt_grad = torch.autograd.grad(tlx.reduce_sum(model_dict[name]), p_t, create_graph=True)[0]
        #     grad = t_grad * grad_temp[name][p_name]
        #     t_model_dict[p_name] = p_t - configs['my_t_lr'] * grad

        # t_model.load_state_dict(t_model_dict)
        zero_grad(t_model)
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['model_name'] in ['GCN', 'LogReg', 'MLP']:
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    if configs['model_name'] in ['APPNP']:
        logits = model(graph.x)
    elif configs['model_name'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    elif configs['model_name'] in ['GraphSAGE', 'SGC']:
        logits = model(graph.x)
    logp = tlx.ops.softmax(logits)
    acc_val = accuracy(logp[my_val_mask], graph.y[my_val_mask])
    loss_val = 0
    print(
        'Epoch %d | acc_train: %.4f| acc_val: %.4f | acc_same: %.4f | hard_loss: %.4f | soft_loss: %.4f| distill_loss: %.4f | evaluate_loss: %.4f' % (
            epoch, acc_train.item(), acc_val.item(), acc_same.item(), hard_loss.item(), soft_loss.item(), train_loss.item(),
            t_train_loss.item()))
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
    cnt = 0
    epoch = 1
    teacher_softmax = tlx.ops.softmax(teacher_logits, axis=1)
    # print(teacher_logits[0],teacher_softmax[0])
    # #
    # teacher_softmax = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.1, 0.9]])
    # t = tlx.reduce_sum(teacher_softmax[0])
    edge_index_no_self_loops, _ = remove_self_loops(graph.edge_index)
    ################################
    # edge_index_no_self_loops = torch.tensor([[0,0,1,1,1,2,2,2,3,3],[1,2,0,2,3,0,1,3,1,2]])
    msg = tlx.gather(teacher_softmax, edge_index_no_self_loops[1])
    nei_logits_sum = unsorted_segment_sum(msg, edge_index_no_self_loops[0], graph.num_nodes)
    # nei_logits_sum = unsorted_segment_sum(msg, edge_index_no_self_loops[0], 4)
    # # PyTorch 示例
    # data = torch.tensor([[1, 2, 3, 4, 5, 6],[12, 2, 3, 4, 5, 6],[15, 2, 3, 4, 5, 6]])
    # segment_ids = torch.tensor([1, 0, 1, 1, 2, 2])
    # msg = tlx.gather(data, segment_ids)
    # num_segments = 3  # 分段的数量
    #
    # # 使用 unsorted_segment_sum 对数据进行分段求和
    # result = unsorted_segment_sum(data, segment_ids, num_segments)
    nei_num = unsorted_segment_sum(tlx.ops.ones_like(edge_index_no_self_loops[0]), edge_index_no_self_loops[0],
                                   graph.num_nodes)  # 分段统计
    # nei_num = unsorted_segment_sum(tlx.ops.ones_like(edge_index_no_self_loops[0]), edge_index_no_self_loops[0],
    #                                                               4)  # 分段统计
    # 总的边数
    # print(tlx.reduce_sum(nei_num))
    nei_probability = tlx.ops.transpose(tlx.ops.divide(tlx.ops.transpose(nei_logits_sum), nei_num))  # 邻居汇总/邻居个数
    # su = tlx.reduce_sum(nei_probability[0])
    nei_entropy = -tlx.ops.reduce_sum(nei_probability * tlx.ops.log(nei_probability), axis=1)
    for i in range(len(nei_entropy)):
        if nei_entropy[i] != nei_entropy[i]:
            nei_entropy[i] = 0.0001

    t_model = MLP(num_layers=2, in_channels=dataset.num_classes + 2, hidden_channels=64, out_channels=1, dropout=0.4,
                  norm=None)
    # print(t_model)
    temp = 0
    best = 0
    train_mask, val_mask, my_val_mask = split_train_mask(graph, dataset.num_classes)
    # # 统计每个掩码中1的数量
    # train_mask_count = torch.sum(train_mask)
    # val_mask_count = torch.sum(val_mask)
    # my_val_mask_count = torch.sum(my_val_mask)
    # test_mask_count = torch.sum(graph.test_mask)
    #
    # # 统计每个掩码对应的标签的数量
    # train_label_count = torch.bincount(graph.y[train_mask.bool()])
    # val_label_count = torch.bincount(graph.y[val_mask.bool()])
    # my_val_label_count = torch.bincount(graph.y[my_val_mask.bool()])
    # test_val_label_count = torch.bincount(graph.y[graph.test_mask.bool()])
    #
    # print(train_label_count, val_label_count, my_val_label_count, test_val_label_count, train_mask_count,
    #       val_mask_count, my_val_mask_count, test_mask_count)
    while epoch < configs['max_epoch']:
        acc_val, loss_val = distill_train(epoch, model, configs, graph, nei_entropy, t_model,
                                          teacher_logits, train_mask, val_mask, my_val_mask, dataset)
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
