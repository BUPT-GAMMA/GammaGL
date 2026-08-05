import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import argparse
import tensorlayerx as tlx
from gammagl.models.unimp import Unimp
from gammagl.datasets import Planetoid
from gammagl.utils import  mask_to_index
from tensorlayerx.model import TrainOneStep, WithLoss

class CrossEntropyLoss(WithLoss):
    def __init__(self, model, loss_func):
        super(CrossEntropyLoss, self).__init__(model,loss_func)

    def forward(self, data, label):
        out = self.backbone_network(data['x'], data['edge_index'])
        out = tlx.gather(out, data['val_idx'])
        label = tlx.reshape(tlx.gather(label, data['val_idx']),shape=(-1,))
        #print(out[0])
        #print(label[0])
        loss = self._loss_fn(out, label)
        return loss


def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst

def get_label_mask(label,node,dtype):
    mask=[1 for i in range(node['train_node1'])]+[0 for i in range(node['train_node2'])]
    random.shuffle(mask)
    label_mask=[]
    for i in range(node['train_node']):
        if mask[i]==0:
            label_mask.append([-1])
        else:
            label_mask.append([(int)(label[i])])
    label_mask+=[[0] for i in range(node['num_node']-node['train_node'])]
    return tlx.ops.convert_to_tensor(label_mask,dtype=dtype)

def merge_feature_label(label,feature):
    return tlx.ops.concat([label,feature],axis=1)

def main(args):
    dataset = Planetoid(root='./',name=args.dataset)
    graph=dataset[0]
    feature=graph.x
    edge_index=graph.edge_index
    label=graph.y
    train_node=int(graph.num_nodes * 0.3)
    train_node1=int(graph.num_nodes * 0.1)
    node = {
        'train_node': train_node,
        'train_node1': train_node1,
        'train_node2': train_node-train_node1,
        'num_node': graph.num_nodes
    }
    val_mask = tlx.ops.concat(
        [tlx.ops.zeros((train_node, 1),dtype=tlx.int32),
        tlx.ops.ones((train_node-train_node1, 1),dtype=tlx.int32)],axis=0)
    test_mask=graph.test_mask
    model=Unimp(dataset)
    loss = tlx.losses.softmax_cross_entropy_with_logits
    optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=5e-4)
    train_weights = model.trainable_weights
    loss_func = CrossEntropyLoss(model, loss)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)
    val_idx = mask_to_index(val_mask)
    test_idx = mask_to_index(test_mask)
    metrics = tlx.metrics.Accuracy()
    data = {
        "x": feature,
        "y": label,
        "edge_index": edge_index,
        "val_idx":val_idx,
        "test_idx": test_idx,
        "num_nodes": graph.num_nodes,
    }

    epochs=args.epochs
    best_val_acc=0
    for epoch in range(epochs):
        model.set_train()
        label_mask=get_label_mask(label,node,feature[0].dtype)
        data['x']=merge_feature_label(label_mask,feature)
        train_loss = train_one_step(data, graph.y)

        model.set_eval()
        logits = model(data['x'], data['edge_index'])
        test_logits = tlx.gather(logits, data['test_idx'])
        test_y = tlx.gather(data['y'], data['test_idx'])
        test_acc = calculate_acc(test_logits, test_y, metrics)

        print("Epoch [{:0>3d}]  ".format(epoch + 1)
              + "   train loss: {:.4f}".format(train_loss.item())
              + "   val acc: {:.4f}".format(test_acc))

        # save best model on evaluation set
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            model.save_weights('./'+ 'unimp' + ".npz", format='npz_dict')
    print("The Best ACC : {:.4f}".format(best_val_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epoch")
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    args = parser.parse_args()
    main(args)