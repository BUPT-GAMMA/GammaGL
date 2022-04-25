import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv.gin_conv import GINConv
from gammagl.layers.pool.glob import global_mean_pool, global_max_pool, global_sum_pool


class GINModel(nn.Module):
	def __init__(self, name=None, input_dim=None, hidden=None, out_dim=None):

		super().__init__(name=name)
		self.conv1 = GINConv(
			nn.Sequential([
				nn.Linear(in_features=input_dim, out_features=hidden, act=nn.activation.ReLU,
						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU,
						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
				nn.BatchNorm1d(num_features=hidden)]
			), learn_eps=False)
		# for i in range(num_layers - 1):
		self.conv2 = GINConv(
			nn.Sequential([
				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU),
				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU),
				nn.BatchNorm1d(num_features=hidden)]
			), learn_eps=False)

		self.lin1 = nn.Linear(in_features=hidden, out_features=hidden)
		self.lin2 = nn.Linear(in_features=hidden, out_features=out_dim)
		self.dropout = tlx.nn.Dropout(p=0.2)

	def reset_parameters(self):
		self.conv1.reset_parameters()
		for conv in self.convs:
			conv.reset_parameters()
		self.lin1.reset_parameters()
		self.lin2.reset_parameters()

	def forward(self, x, edge_index, batch):
		x = self.conv1(x, edge_index)
		x = self.conv2(x, edge_index)
		x = global_mean_pool(x, batch)
		x = nn.activation.ReLU()(self.lin1(x))
		x = self.dropout(x)
		x = self.lin2(x)
		return x

