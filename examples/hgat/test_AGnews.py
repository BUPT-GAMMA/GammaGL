import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['TL_BACKEND'] = 'tensorflow'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from agnews import AGNewsDataset
from gammagl.datasets import IMDB
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
import gammagl.transforms as T
from gammagl.utils import mask_to_index
from hgat import HGAT
import torch
import random
import numpy as np

def set_seed(seed=0):
    random.seed(seed)
    tlx.set_seed(0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(0)

class SemiSpvzLoss(WithLoss):
    def __init__(self, net, loss_fn):
        super(SemiSpvzLoss, self).__init__(backbone=net, loss_fn=loss_fn)

    def forward(self, data, y):
        logits = self.backbone_network(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
        train_logits = tlx.gather(logits['text'], data['train_idx'])
        train_y = tlx.gather(data['y'], data['train_idx'])
        loss = self._loss_fn(train_logits, train_y)
        return loss
    
def calculate_acc(logits, y, metrics):
    metrics.update(logits, y)
    rst = metrics.result()
    metrics.reset()
    return rst


dataset = IMDB(r'')
graph = dataset[0]
print(graph['movie'].y)
dataset = AGNewsDataset(r'')
graph = dataset[0]
y = tlx.argmax(graph['text'].y,axis=1)
print(graph['text'].y[0:1000])
print(y[0:1000])
# for mindspore, it should be passed into node indices
train_idx = mask_to_index(graph['text'].train_mask,)
test_idx = mask_to_index(graph['text'].test_mask)
val_idx = mask_to_index(graph['text'].val_mask)


in_channel = {'text':5126, 'topic':4962, 'entity': 4378}
num_nodes_dict = {'text': 3200, 'topic': 15, 'entity': 5677}

print(np.unique(graph['text'].y))
net = HGAT(
    in_channels=in_channel,
    out_channels=4, # graph.num_classes,
    metadata=graph.metadata(),
    drop_rate=0.5,
    hidden_channels=64,
    name = 'hgat',
)


optimizer = tlx.optimizers.Adam(lr=0.01, weight_decay=1e-3)
metrics = tlx.metrics.Accuracy()
train_weights = net.trainable_weights

loss_func = tlx.losses.softmax_cross_entropy_with_logits
semi_spvz_loss = SemiSpvzLoss(net, loss_func)
train_one_step = TrainOneStep(semi_spvz_loss, optimizer, train_weights)

data = {
    "x_dict": graph.x_dict,
    "y":y,
    "edge_index_dict": graph.edge_index_dict,
    "train_idx": train_idx,
    "test_idx": test_idx,
    "val_idx": val_idx,
    "num_nodes_dict": num_nodes_dict,
}

best_val_acc = 0
n_epoch = 200
for epoch in range(n_epoch):
    net.set_train()
    train_loss = train_one_step(data, y)
    net.set_eval()
    logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
    val_logits = tlx.gather(logits['text'], data['val_idx'])
    val_y = tlx.gather(data['y'], data['val_idx'])
    val_acc = calculate_acc(val_logits, val_y, metrics)

    print("Epoch [{:0>3d}]  ".format(epoch + 1)
            + "   train_loss: {:.4f}".format(train_loss.item())
            # + "   train_acc: {:.4f}".format(train_acc)
            + "   val_acc: {:.4f}".format(val_acc))

    # save best model on evaluation set
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        net.save_weights("./ggl_" + net.name + '_model' + ".npz", format='npz_dict')

net.load_weights("./ggl_" + net.name + '_model' + ".npz", format='npz_dict')
net.set_eval()
logits = net(data['x_dict'], data['edge_index_dict'], data['num_nodes_dict'])
test_logits = tlx.gather(logits['movie'], data['test_idx'])
test_y = tlx.gather(data['y'], data['test_idx'])
test_acc = calculate_acc(test_logits, test_y, metrics)
print("Test acc:  {:.4f}".format(test_acc))

import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open("output.txt", "w") as f:
    f.write("Current Time: " + current_time + "\n\n")
    f.write("GGL:"+ str(dataset) + "\n")
    f.write("Test acc:  {:.4f}\n\n".format(test_acc))



def npz2pth_with_mapping(network, npz_path, pth_path):
    npz_dict = np.load(npz_path)
    torch_dict = network.state_dict()

    # 参数映射字典：npz 文件中的参数名 -> PyTorch 模型中的参数名
    name_mapping = {
        'linear_1/weights:0':'hin_conv.gcn_dict.movie.lin.weight',
        'linear_2/weights:0':'hin_conv.gcn_dict.director.lin.weight',
        'linear_3/weights:0':'hin_conv.gcn_dict.actor.lin.weight',
        'linear_4/weights:0':'hgat_conv.Linear_dict_l.movie.weight',
        'linear_6/weights:0':'hgat_conv.Linear_dict_l.director.weight',
        'linear_8/weights:0':'hgat_conv.Linear_dict_l.actor.weight',
        'linear_5/weights:0':'hgat_conv.Linear_dict_r.movie.weight',
        'linear_7/weights:0':'hgat_conv.Linear_dict_r.director.weight',
        'linear_9/weights:0':'hgat_conv.Linear_dict_r.actor.weight',
        'linear_10/weights:0':'hgat_conv.Linear_l.weight',
        'linear_11/weights:0':'hgat_conv.Linear_r.weight',
        'linear_12/weights:0':'linear.weight',
        'linear_12/biases:0':'linear.bias'
    }

    for npz_key, torch_key in name_mapping.items():

        if npz_key in npz_dict:
            param_shape = torch_dict[torch_key].shape
            npz_param = npz_dict[npz_key]

            if npz_param.shape == param_shape:
                torch_dict[torch_key] = torch.from_numpy(npz_param).float()
            elif npz_param.T.shape == param_shape:
                print(f"信息: 对参数 '{npz_key}' 进行转置以匹配 PyTorch 形状。")
                torch_dict[torch_key] = torch.from_numpy(npz_param.T).float()
            else:
                print(f"错误: 参数 '{npz_key}' 的形状 {npz_param.shape} 与 PyTorch 形状 {param_shape} 不匹配。")
        else:
            print(f"警告: npz 文件中缺少与 PyTorch 参数 '{torch_key}' 对应的参数 '{npz_key}'。")

    torch.save(torch_dict, pth_path)
    print("转换完成。")

npz_path = './ggl_hgat_model.npz'  # 这是 GGL 模型的参数文件
pth_path = './pyg_hgat_model.pth'  # 转换后的 PyTorch 参数文件

pyg_model = PyG_HGATModel(in_channels=in_channel,
                        out_channels=len(np.unique(graph['movie'].y)), # graph.num_classes,
                        metadata=graph.metadata(),
                        drop_rate=0.5,
                        hidden_channels=512,
                        )  # 根据实际情况初始化 PyG 模型

# 加载 .npz 文件
npz_file = np.load('ggl_hgat_model.npz')

# # 打印所有参数名称
# print("Parameters in npz file:")
# for param in npz_file:
#     print(param)

# print("Parameters in PyTorch model:")
# for name, param in pyg_model.named_parameters():
#     print(name)

def convert_to_pytorch(x_dict):
    for key, value in x_dict.items():
        # print(value)
        x_dict[key] = torch.tensor(tlx.convert_to_numpy(value))

    return x_dict

npz2pth_with_mapping(pyg_model, npz_path, pth_path)
pyg_model.load_state_dict(torch.load(pth_path))


pyg_model.eval()
with torch.no_grad():
    out = pyg_model(convert_to_pytorch(data['x_dict']), convert_to_pytorch(data['edge_index_dict']), data['num_nodes_dict'])
    pred = out['movie'].argmax(dim=1)

    test_acc = pred[graph['movie'].test_mask.numpy().tolist()].eq(torch.tensor(tlx.convert_to_numpy(test_y))).sum().item() / torch.tensor(tlx.convert_to_numpy(graph['movie'].test_mask)).sum().item()

# 输出结果
print("PyG Model Test Accuracy: {:.4f}".format(test_acc))
with open("output.txt", "a") as f:
    f.write("PyG:"+ str(dataset) + "\n")
    f.write("Test acc:  {:.4f}\n\n".format(test_acc))

