# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   diffpool_trainer.py
@Time    :   2022/04/05 18:00:00
@Author  :   yang yuxin
"""

import os
os.environ['TL_BACKEND'] = 'torch' # set your backend here, default `tensorflow`
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
sys.path.insert(0, os.path.abspath('../../'))

import pickle
import shutil
import random
import argparse
import tensorlayerx as tlx
import numpy as np
import networkx as nx
import abc
from tensorboardX import SummaryWriter
from tensorlayerx import Variable #from torch.autograd import Variable
import sklearn.metrics as metrics

from gammagl.datasets.dd import prepare_val_data,read_graphfile
from gammagl.models.diffpool import GcnEncoderGraph#,GcnSet2SetEncoder,SoftPoolingGcnEncoder
from gammagl.data.graph_sampler import GraphSampler


def evaluate(dataset, model, args, name='Validation', max_num_examples=None):
    model.eval()
    labels = []
    preds = []
    for batch_idx, data in enumerate(dataset):
        adj = Variable(data['adj'].float())#.cuda() #requires_grad=False
        h0 = Variable(data['feats'].float())#.cuda()
        labels.append(data['label'].long().numpy())
        batch_num_nodes = data['num_nodes'].int().numpy()
        assign_input = Variable(data['assign_feats'].float())#.cuda()
                        #Variable(data['assign_feats'].float(), requires_grad=False).cuda()
        ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input) #20,2
        indices = tlx.argmax(ypred, axis=1) #_, indices = torch.max(ypred, 1) #20
        preds.append(indices.cpu().data.numpy())

        if max_num_examples is not None:
            if (batch_idx + 1) * args.batch_size > max_num_examples:
                break

    #6*20 --> 120
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    # labels=tlx.convert_to_tensor(labels) #6*20=120
    # preds=tlx.convert_to_tensor(preds) #6*20=120


    # metrics = tlx.metrics.Accuracy()
    # metrics.update(preds, labels)
    # result = metrics.result()
    #print(name, " accuracy:", result)
    # metrics.reset()

    result = {
        # 'prec': metrics.precision_score(labels, preds, average='macro'),
        #       'recall': metrics.recall_score(labels, preds, average='macro'),
              'acc': metrics.accuracy_score(labels, preds)
        #,'F1': metrics.f1_score(labels, preds, average="micro")
        }
    print(name, " accuracy:", result['acc'])

    return result

def gen_prefix(args):
    if args.bmname is not None:
        name = args.bmname
    else:
        name = args.dataset
    name += '_' + args.method
    if args.method == 'soft-assign':
        name += '_l' + str(args.num_gc_layers) + 'x' + str(args.num_pool)
        name += '_ar' + str(int(args.assign_ratio*100))
        if args.linkpred:
            name += '_lp'
    else:
        name += '_l' + str(args.num_gc_layers)
    name += '_h' + str(args.hidden_dim) + '_o' + str(args.output_dim)
    if not args.bias:
        name += '_nobias'
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def train(dataset, model, args, same_feat=True, val_dataset=None, test_dataset=None, writer=None,
          mask_nodes=True):
    writer_batch_idx = [0, 3, 6, 9]

    optimizer = tlx.optimizers.Adam(lr=0.001)
            #参数：(filter(lambda p: p.requires_grad, model.parameters()), learning_rate=0.001)
    iter = 0
    # best_val_result = {
    #     'epoch': 0,
    #     'loss': 0,
    #     'acc': 0}
    # test_result = {
    #     'epoch': 0,
    #     'loss': 0,
    #     'acc': 0}
    train_accs = []
    train_epochs = []
    # best_val_accs = []
    # best_val_epochs = []
    test_accs = []
    # test_epochs = []
    val_accs = []

    for epoch in range(args.num_epochs):
        total_time = 0
        avg_loss = 0.0
        #model.train()
        model.set_train()
        print('Epoch: ', epoch)
        for batch_idx, data in enumerate(dataset):
            # adj:20,1000,1000; feat 20,1000,89; assign_feats:20,1000,89; label:20
            # begin_time = time.time()
            model.zero_grad()
            adj = Variable(data['adj'].float(),"adj",False)#.cuda() #,requires_grad=False
            h0 = Variable(data['feats'].float(),"h0",False)#.cuda()#,requires_grad=False
            label = Variable(data['label'].long(),"label",False)#.cuda()
            batch_num_nodes = data['num_nodes'].int().numpy() if mask_nodes else None
            assign_input = Variable(data['assign_feats'].float(),"assign_input",False)#.cuda()#,requires_grad=False


            ypred = model(h0, adj, batch_num_nodes, assign_x=assign_input)  # 20,2
            if not args.method == 'soft-assign' or not args.linkpred:
                loss = model.loss(ypred, label)
            else:
                loss = model.loss(ypred, label, adj, batch_num_nodes)
            #loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            #optimizer.step()
            g=optimizer.gradient(loss,weights=model.trainable_weights)
            optimizer.apply_gradients(zip(g,model.trainable_weights))

            iter += 1
            avg_loss += loss
            # if iter % 20 == 0:
            #    print('Iter: ', iter, ', loss: ', loss.data[0])

        avg_loss /= batch_idx + 1
        if writer is not None:
            writer.add_scalar('loss/avg_loss', avg_loss, epoch)
            if args.linkpred:
                writer.add_scalar('loss/linkpred_loss', model.link_loss, epoch)
        print('Avg loss: ', avg_loss)
        result = evaluate(dataset, model, args, name='Train', max_num_examples=100)
        train_accs.append(result['acc'])
        train_epochs.append(epoch)
        if val_dataset is not None:
            val_result = evaluate(val_dataset, model, args, name='Validation')
            val_accs.append(val_result['acc'])
        # if val_result['acc'] > best_val_result['acc'] - 1e-7:
        #     best_val_result['acc'] = val_result['acc']
        #     best_val_result['epoch'] = epoch
        #     best_val_result['loss'] = avg_loss
        if test_dataset is not None:
            test_result = evaluate(test_dataset, model, args, name='Test')
            test_result['epoch'] = epoch
        if writer is not None:
            writer.add_scalar('acc/train_acc', result['acc'], epoch)
            writer.add_scalar('acc/val_acc', val_result['acc'], epoch)
            # writer.add_scalar('loss/best_val_loss', best_val_result['loss'], epoch)
            if test_dataset is not None:
                writer.add_scalar('acc/test_acc', test_result['acc'], epoch)
        # print('Best val result: ', best_val_result)
        # best_val_epochs.append(best_val_result['epoch'])
        # best_val_accs.append(best_val_result['acc'])
        # if test_dataset is not None:
        #     print('Test result: ', test_result)
        #     test_epochs.append(test_result['epoch'])
        #     test_accs.append(test_result['acc'])
    return model, val_accs


def prepare_data(graphs, args, test_graphs=None, max_nodes=0):
    random.shuffle(graphs)
    if test_graphs is None:
        train_idx = int(len(graphs) * args.train_ratio)
        test_idx = int(len(graphs) * (1 - args.test_ratio))#args
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx: test_idx]
        test_graphs = graphs[test_idx:]
    else:
        train_idx = int(len(graphs) * args.train_ratio)
        train_graphs = graphs[:train_idx]
        val_graphs = graphs[train_idx:]
    # print('Num training graphs: ', len(train_graphs),
    #       '; Num validation graphs: ', len(val_graphs),
    #       '; Num testing graphs: ', len(test_graphs))
    #
    # print('Number of graphs: ', len(graphs))
    # print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    # print('Max, avg, std of graph size: ',
    #       max([G.number_of_nodes() for G in graphs]), ', '
    #                                                   "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
    #       ', '
    #       "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    train_dataset_loader = tlx.dataflow.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    val_dataset_loader = tlx.dataflow.DataLoader( # torch.utils.data.DataLoader
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    dataset_sampler = GraphSampler(test_graphs, normalize=False, max_num_nodes=max_nodes,
                                   features=args.feature_type)
    test_dataset_loader = tlx.dataflow.DataLoader(
        dataset_sampler,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)

    return train_dataset_loader, val_dataset_loader, test_dataset_loader, \
           dataset_sampler.max_num_nodes, dataset_sampler.feat_dim, dataset_sampler.assign_feat_dim


#gen #feat
class FeatureGen(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def gen_node_features(self, G):
        pass

class ConstFeatureGen(FeatureGen):
    def __init__(self, val):
        self.val = val

    def gen_node_features(self, G):
        feat_dict = {i:{'feat': self.val} for i in G.nodes()}
        nx.set_node_attributes(G, feat_dict)

class GaussianFeatureGen(FeatureGen):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def gen_node_features(self, G):
        feat = np.random.multivariate_normal(self.mu, self.sigma, G.number_of_nodes())
        feat_dict = {i:{'feat': feat[i]} for i in range(feat.shape[0])}
        nx.set_node_attributes(G, feat_dict)

#data
def gen_ba(n_range, m_range, num_graphs, feature_generator=None):
    graphs = []
    for i in np.random.choice(n_range, num_graphs):
        for j in np.random.choice(m_range, 1):
            graphs.append(nx.barabasi_albert_graph(i,j))

    if feature_generator is None:
        feature_generator = ConstFeatureGen(0) #featgen.
    for G in graphs:
        feature_generator.gen_node_features(G)
    return graphs

def gen_er(n_range, p, num_graphs, feature_generator=None):
    graphs = []
    for i in np.random.choice(n_range, num_graphs):
        graphs.append(nx.erdos_renyi_graph(i,p))

    if feature_generator is None:
        feature_generator = ConstFeatureGen(0) #featgen.
    for G in graphs:
        feature_generator.gen_node_features(G)
    return graphs

def gen_2community_ba(n_range, m_range, num_graphs, inter_prob, feature_generators):
    ''' Each community is a BA graph.
    Args:
        inter_prob: probability of one node connecting to any node in the other community.
    '''

    if feature_generators is None:
        mu0 = np.zeros(10)
        mu1 = np.ones(10)
        sigma0 = np.ones(10, 10) * 0.1
        sigma1 = np.ones(10, 10) * 0.1
        fg0 = GaussianFeatureGen(mu0, sigma0) #featgen.
        fg1 = GaussianFeatureGen(mu1, sigma1) #featgen.
    else:
        fg0 = feature_generators[0]
        fg1 = feature_generators[1] if len(feature_generators) > 1 else feature_generators[0]

    graphs1 = []
    graphs2 = []
    #for (i1, i2) in zip(np.random.choice(n_range, num_graphs),
    #                    np.random.choice(n_range, num_graphs)):
    #    for (j1, j2) in zip(np.random.choice(m_range, num_graphs),
    #                        np.random.choice(m_range, num_graphs)):
    graphs0 = gen_ba(n_range, m_range, num_graphs, fg0)
    graphs1 = gen_ba(n_range, m_range, num_graphs, fg1)
    graphs = []
    for i in range(num_graphs):
        G = nx.disjoint_union(graphs0[i], graphs1[i])
        n0 = graphs0[i].number_of_nodes()
        for j in range(n0):
            if np.random.rand() < inter_prob:
                target = np.random.choice(G.number_of_nodes() - n0) + n0
                G.add_edge(j, target)
        graphs.append(G)
    return graphs

def gen_2hier(num_graphs, num_clusters, n, m_range, inter_prob1, inter_prob2, feat_gen):
    ''' Each community is a BA graph.
    Args:
        inter_prob1: probability of one node connecting to any node in the other community within
            the large cluster.
        inter_prob2: probability of one node connecting to any node in the other community between
            the large cluster.
    '''
    graphs = []

    for i in range(num_graphs):
        clusters2 = []
        for j in range(len(num_clusters)):
            clusters = gen_er(range(n, n+1), 0.5, num_clusters[j], feat_gen[0])
            G = nx.disjoint_union_all(clusters)
            for u1 in range(G.number_of_nodes()):
                if np.random.rand() < inter_prob1:
                    target = np.random.choice(G.number_of_nodes() - n)
                    # move one cluster after to make sure it's not an intra-cluster edge
                    if target // n >= u1 // n:
                        target += n
                    G.add_edge(u1, target)
            clusters2.append(G)
        G = nx.disjoint_union_all(clusters2)
        cluster_sizes_cum = np.cumsum([cluster2.number_of_nodes() for cluster2 in clusters2])
        curr_cluster = 0
        for u1 in range(G.number_of_nodes()):
            if u1 >= cluster_sizes_cum[curr_cluster]:
                curr_cluster += 1
            if np.random.rand() < inter_prob2:
                target = np.random.choice(G.number_of_nodes() -
                        clusters2[curr_cluster].number_of_nodes())
                # move one cluster after to make sure it's not an intra-cluster edge
                if curr_cluster == 0 or target >= cluster_sizes_cum[curr_cluster - 1]:
                    target += cluster_sizes_cum[curr_cluster]
            G.add_edge(u1, target)
        graphs.append(G)

    return graphs

def syn_community1v2(args, writer=None, export_graphs=False):
    graphs1 = gen_ba(range(40, 60), range(4, 5), 500,#datagen.
                             ConstFeatureGen(np.ones(args.input_dim, dtype=float)))#featgen.
    for G in graphs1:
        G.graph['label'] = 0

    graphs2 = gen_2community_ba(range(20, 30), range(4, 5), 500, 0.3, #datagen.
                                        [ConstFeatureGen(np.ones(args.input_dim, dtype=float))])#featgen.
    for G in graphs2:
        G.graph['label'] = 1

    graphs = graphs1 + graphs2

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        # model = SoftPoolingGcnEncoder(
        #     max_num_nodes,
        #     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #     bn=args.bn, linkpred=args.linkpred, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
         print('Method: base-set2set')
        # model = GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
        #                                    args.num_gc_layers, bn=args.bn).cuda()
    else:
        #print('Method: base')
        model = GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn)#.cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def syn_community2hier(args, writer=None):
    # data
    feat_gen = [ConstFeatureGen(np.ones(args.input_dim, dtype=float))]#featgen
    graphs1 = gen_2hier(1000, [2, 4], 10, range(4, 5), 0.1, 0.03, feat_gen) #datagen.
    graphs2 = gen_2hier(1000, [3, 3], 10, range(4, 5), 0.1, 0.03, feat_gen) #datagen.
    graphs3 = gen_2community_ba(range(28, 33), range(4, 7), 1000, 0.25, feat_gen) #datagen.

    for G in graphs1:
        G.graph['label'] = 0
    for G in graphs2:
        G.graph['label'] = 1
    for G in graphs3:
        G.graph['label'] = 2

    graphs = graphs1 + graphs2 + graphs3

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = prepare_data(graphs, args)

    if args.method == 'soft-assign':
        print('Method: soft-assign')
        # model = SoftPoolingGcnEncoder(
        #     max_num_nodes,
        #     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #     bn=args.bn, linkpred=args.linkpred, args=args, assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
         print('Method: base-set2set')
        # model = GcnSet2SetEncoder(input_dim, args.hidden_dim, args.output_dim, 2,
        #                                    args.num_gc_layers, bn=args.bn, args=args,
        #                                    assign_input_dim=assign_input_dim).cuda()
    else:
        print('Method: base')
        model = GcnEncoderGraph(input_dim, args.hidden_dim, args.output_dim, 2,
                                         args.num_gc_layers, bn=args.bn, args=args)#.cuda()
    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)


def pkl_task(args, feat=None):
    with open(os.path.join(args.datadir, args.pkl_fname), 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    graphs = data[0]
    labels = data[1]
    test_graphs = data[2]
    test_labels = data[3]

    for i in range(len(graphs)):
        graphs[i].graph['label'] = labels[i]
    for i in range(len(test_graphs)):
        test_graphs[i].graph['label'] = test_labels[i]

    if feat is None:
        featgen_const = ConstFeatureGen(np.ones(args.input_dim, dtype=float)) #featgen
        for G in graphs:
            featgen_const.gen_node_features(G)
        for G in test_graphs:
            featgen_const.gen_node_features(G)

    train_dataset, test_dataset, max_num_nodes = prepare_data(graphs, args, test_graphs=test_graphs)
    model = GcnEncoderGraph(
        args.input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        args.num_gc_layers, bn=args.bn).cuda()
    train(train_dataset, model, args, test_dataset=test_dataset)
    evaluate(test_dataset, model, args, 'Validation')


def benchmark_task(args, writer=None, feat='node-label'):
    graphs = read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in graphs[0].node[0]:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                G.node[u]['feat'] = np.array(G.node[u]['label'])
    else:
        print('Using constant labels')
        featgen_const = ConstFeatureGen(np.ones(args.input_dim, dtype=float)) #featgen.
        for G in graphs:
            featgen_const.gen_node_features(G)

    train_dataset, val_dataset, test_dataset, max_num_nodes, input_dim, assign_input_dim = \
        prepare_data(graphs, args, max_nodes=args.max_nodes)
    if args.method == 'soft-assign':
        print('Method: soft-assign')
        # model = SoftPoolingGcnEncoder(
        #     max_num_nodes,
        #     input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #     args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #     bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        #     assign_input_dim=assign_input_dim).cuda()
    elif args.method == 'base-set2set':
         print('Method: base-set2set')
        # model = GcnSet2SetEncoder(
        #     input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        #     args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
    else:
        # print('Method: base')
        model = GcnEncoderGraph(
            input_dim, args.hidden_dim, args.output_dim, args.num_classes,
            args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()

    train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=test_dataset,
          writer=writer)
    evaluate(test_dataset, model, args, 'Validation')

#import util
# ---- NetworkX compatibility
def node_iter(G):
    # if float(nx.__version__)<2.0:
    #     return G.nodes()
    # else:
    return G.nodes

def node_dict(G):
    #if float(nx.__version__)>2.1:
    node_dict = G.nodes #159
    # else:
    #     node_dict = G.node
    return node_dict
# ---------------------------

def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a

def benchmark_task_val(args, writer=None, feat='node-label'):
    all_vals = []
    graphs = read_graphfile(args.datadir, args.bmname, max_nodes=args.max_nodes)

    example_node = node_dict(graphs[0])[0]

    if feat == 'node-feat' and 'feat_dim' in graphs[0].graph:
        print('Using node features')
        input_dim = graphs[0].graph['feat_dim']
    elif feat == 'node-label' and 'label' in example_node:
        print('Using node labels')
        for G in graphs:
            for u in G.nodes():
                node_dict(G)[u]['feat'] = np.array(node_dict(G)[u]['label'])  # ？
    else:
        print('Using constant labels')
        featgen_const = ConstFeatureGen(np.ones(args.input_dim, dtype=float)) #featgen.
        for G in graphs:
            featgen_const.gen_node_features(G)

    for i in range(10):
        train_dataset, val_dataset, max_num_nodes, input_dim, assign_input_dim = \
            prepare_val_data(graphs, args, i, max_nodes=args.max_nodes)
        if args.method == 'base':
            print('Method: base')
            model = GcnEncoderGraph(
                 input_dim, args.hidden_dim, args.output_dim, args.num_classes,
                 args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args)#.cuda()
        # if args.method == 'soft-assign':
        #      print('Method: soft-assign')
        #     model = SoftPoolingGcnEncoder(
        #         max_num_nodes,
        #         input_dim, args.hidden_dim, args.output_dim, args.num_classes, args.num_gc_layers,
        #         args.hidden_dim, assign_ratio=args.assign_ratio, num_pooling=args.num_pool,
        #         bn=args.bn, dropout=args.dropout, linkpred=args.linkpred, args=args,
        #         assign_input_dim=assign_input_dim).cuda()
        # elif args.method == 'base-set2set':
        #      print('Method: base-set2set')
        #      model = GcnSet2SetEncoder(
        #         input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        #         args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()
        #  else:
        #      print('Method: base')
        #      model = GcnEncoderGraph(
        #          input_dim, args.hidden_dim, args.output_dim, args.num_classes,
        #          args.num_gc_layers, bn=args.bn, dropout=args.dropout, args=args).cuda()


        _, val_accs = train(train_dataset, model, args, val_dataset=val_dataset, test_dataset=None,
                            writer=writer)
        all_vals.append(np.array(val_accs))
    all_vals = np.vstack(all_vals)
    all_vals = np.mean(all_vals, axis=0)
    print(all_vals)
    print(np.max(all_vals))
    print(np.argmax(all_vals))


def main(prog_args):
    import multiprocessing as mp
    mp.set_start_method('spawn')
    # export scalar data to JSON for external processing
    path = os.path.join(prog_args.logdir, gen_prefix(prog_args))
    if os.path.isdir(path):
        # print('Remove existing log dir: ', path)
        shutil.rmtree(path)
    writer = SummaryWriter(path)
    writer = None

    # print('CUDA', prog_args.cuda)

    if prog_args.bmname is not None:
        benchmark_task_val(prog_args, writer=writer)
    elif prog_args.pkl_fname is not None:
        pkl_task(prog_args)
    elif prog_args.dataset is not None:
        if prog_args.dataset == 'syn1v2':
            syn_community1v2(prog_args, writer=writer)
        if prog_args.dataset == 'syn2hier':
            syn_community2hier(prog_args, writer=writer)

    writer.close()


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', default='syn1v2',dest='dataset',
                           help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
                                  help='Name of the benchmark dataset')
    io_parser.add_argument('--pkl', dest='pkl_fname',
                           help='Name of the pkl data file')

    softpool_parser = parser.add_argument_group()
    softpool_parser.add_argument('--assign-ratio',default=0.1, dest='assign_ratio', type=float,
                                 help='ratio of number of nodes in consecutive layers')
    softpool_parser.add_argument('--num-pool', default=1,dest='num_pool', type=int,
                                 help='number of pooling layers')
    parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                        const=True, default=False,
                        help='Whether link prediction side objective is used')

    parser.add_argument('--datadir', dest='datadir', default='data',
                        help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir', default='log',
                        help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda', default='8',
                        help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,default=1000,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
                        help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=20,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,default=0.8,
                        help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,default=1,
                        help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',default='default',
                        help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int, default=10,
                        help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=20,
                        help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,default=20,
                        help='Output dimension')
    parser.add_argument('--num-classes', dest='num_classes', type=int,default=2,
                        help='Number of label classes')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,default=3,
                        help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
                        const=False, default=True,
                        help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0,
                        help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
                        const=False, default=True,
                        help='Whether to add bias. Default to True.')
    parser.add_argument('--no-log-graph', dest='log_graph', action='store_const',
                        const=False, default=True,
                        help='Whether disable log graph')

    parser.add_argument('--method', dest='method',default='base',
                        help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',default='',
                        help='suffix added to the output filename')
    parser.set_defaults(datadir='data',
                        logdir='log',
                        dataset='syn1v2',
                        max_nodes=1000,
                        cuda='8',
                        feature_type='default',
                        lr=0.001,
                        clip=2.0,
                        batch_size=20,
                        num_epochs=50,
                        train_ratio=0.8,
                        test_ratio=0.1,
                        num_workers=1,
                        input_dim=10,
                        hidden_dim=20,
                        output_dim=20,
                        num_classes=2,
                        num_gc_layers=3,
                        dropout=0.0,
                        method='base',
                        name_suffix='',
                        assign_ratio=0.1,
                        num_pool=1
                        )
    prog_args = parser.parse_args()
    main(prog_args)
