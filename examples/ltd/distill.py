import tensorlayerx as tlx
from gammagl.utils import add_self_loops, remove_self_loops
from gammagl.mpops import *
from gammagl.models import GCNModel, GATModel, MLP
import torch
torch.autograd.set_detect_anomaly(True)


def layer_normalize(logits):
    mean = tlx.ops.reduce_mean(logits, axis=1, keepdims=True)
    std = tlx.ops.reduce_std(logits, axis=1, keepdims=True)
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


def choose_model(configs):
    if configs['student'] == 'GCN':
        model = GCNModel(feature_dim=configs['num_node_features'],
                         hidden_dim=configs['hidden_dim'],
                         num_class=configs['num_classes'],
                         drop_rate=configs['drop_rate'],
                         num_layers=configs['num_layers'],
                         norm='both',
                         name='GCN')
    elif configs['student'] == 'GAT':
        model = GATModel(feature_dim=configs['num_node_features'],
                         hidden_dim=configs['hidden_dim'],
                         num_class=configs['num_classes'],
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


def distill_train(epoch, model, configs, graph, nei_entropy, t_model, teacher_logits):
    model.set_train()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['student'] == 'GCN':
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    elif configs['student'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    logits = layer_normalize(logits)
    acc_train = accuracy(logits[configs['train_mask']], graph.y[configs['train_mask']])
    acc_same = accuracy_t(logits, teacher_logits)
    with torch.no_grad():
        f_logits = torch.norm(logits, 2, dim=1)
    extract_x = tlx.ops.concat(
        [logits, tlx.expand_dims(f_logits, axis=1), tlx.expand_dims(nei_entropy, axis=1)],
        axis=1)

    temparature = t_model(extract_x)
    temparature = (tlx.ops.sigmoid(temparature) - 0.2) * configs['k']
    temparature = tlx.ops.where(tlx.ops.abs(temparature) < 0.0001, 0.001 * tlx.ops.ones_like(temparature),
                                temparature)
    teacher_logits_t = tlx.ops.transpose(
        tlx.ops.divide(tlx.ops.transpose(teacher_logits), tlx.ops.squeeze(temparature, axis=1)))
    teacher_softmax = tlx.ops.softmax(teacher_logits_t, axis=1)
    student_hard = tlx.ops.softmax(logits, axis=1)
    labels_one_hot = tlx.nn.OneHot(depth=configs['num_classes'])(graph.y)
    hard_loss = -tlx.ops.reduce_sum(
        (labels_one_hot[configs['train_mask']] + 1e-6) * tlx.ops.log(student_hard[configs['train_mask']] + 1e-6))
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
        logits = layer_normalize(logits)
        student_hard = tlx.ops.softmax(logits, axis=1)
        labels_one_hot = tlx.nn.OneHot(depth=configs['num_classes'])(graph.y)
        t_train_loss = -tlx.ops.reduce_sum(
            (labels_one_hot[configs['val_mask']] + 1e-6) * tlx.ops.log(student_hard[configs['val_mask']] + 1e-6))
        for name, t in model.named_parameters():
            t_grad = torch.autograd.grad(t_train_loss, t, create_graph=True, allow_unused=True)[0]
            t_grad = t_grad.detach()
            model_dict[name] = model_dict[name] * t_grad
        t_model_dict = {}
        for name, t in t_model.named_parameters():
            t_model_dict[name] = t.detach()
        for p_name, p in model.named_parameters():
            for name, t in t_model.named_parameters():
                agr = torch.autograd.grad(tlx.reduce_sum(model_dict[p_name]), t, create_graph=True, allow_unused=True)[
                    0]
                t_model_dict[name] = t_model_dict[name] - configs['my_t_lr'] * agr

        t_model.load_state_dict(t_model_dict)
        zero_grad(t_model)
    model.set_eval()
    edge_index, _ = add_self_loops(graph.edge_index, num_nodes=graph.num_nodes)
    if configs['student'] == 'GCN':
        logits = model(graph.x, edge_index, None, graph.num_nodes)
    elif configs['student'] == 'GAT':
        logits = model(graph.x, edge_index, graph.num_nodes)
    acc_val = accuracy(logits[configs['my_val_mask']], graph.y[configs['my_val_mask']])
    print(
        'Epoch %d | acc_train: %.4f| acc_val: %.4f | acc_same: %.4f | hard_loss: %.4f | soft_loss: %.4f| distill_loss: %.4f | evaluate_loss: %.4f' % (
            epoch, acc_train.item(), acc_val.item(), acc_same.item(), hard_loss.item(), soft_loss.item(),
            train_loss.item(),
            t_train_loss.item()))
    return acc_val, train_loss


def model_train(configs, model, graph, teacher_logits):
    teacher_softmax = tlx.ops.softmax(teacher_logits, axis=1)
    edge_index_no_self_loops, _ = remove_self_loops(graph.edge_index)
    msg = tlx.gather(teacher_softmax, edge_index_no_self_loops[1])
    nei_logits_sum = unsorted_segment_sum(msg, edge_index_no_self_loops[0], graph.num_nodes)
    nei_num = unsorted_segment_sum(tlx.ops.ones_like(edge_index_no_self_loops[0], dtype=torch.float),
                                   edge_index_no_self_loops[0],
                                   graph.num_nodes)
    nei_probability = tlx.ops.transpose(tlx.ops.divide(tlx.ops.transpose(nei_logits_sum), nei_num))
    nei_entropy = -tlx.ops.reduce_sum(nei_probability * tlx.ops.log(nei_probability), axis=1)
    for i in range(len(nei_entropy)):
        if nei_entropy[i] != nei_entropy[i]:
            nei_entropy[i] = 0.0001

    t_model = MLP(num_layers=2, in_channels=configs['num_classes'] + 2, hidden_channels=64, out_channels=1, dropout=0.6,
                  norm=None, act=tlx.nn.ReLU())

    best = 0
    cnt = 0
    epoch = 1
    while epoch < configs['max_epoch']:
        acc_val, train_loss = distill_train(epoch, model, configs, graph, nei_entropy, t_model,
                                            teacher_logits)
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
    acc_test = distill_test(model, graph, configs)
    return acc_test


def distill_test(model, graph, configs):
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
    logp = tlx.ops.log(tlx.ops.softmax(logits))
    acc_test = accuracy(logp[graph.test_mask], graph.y[graph.test_mask])
    return acc_test
