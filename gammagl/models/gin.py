import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv.gin_conv import GINConv
from gammagl.layers.pool.glob import global_mean_pool


class GINModel(nn.Module):
	def __init__(self, name=None, input_dim=None, hidden=None, out_dim=None):

		super().__init__(name=name)
		self.conv1 = GINConv(
			nn.SequentialLayer(
				nn.Dense(in_channels=input_dim, n_units=hidden, act=tlx.ReLU,
						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
				nn.Dense(in_channels=hidden, n_units=hidden, act=tlx.ReLU,
						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
				nn.BatchNorm1d(num_features=hidden),
			), learn_eps=False)
		# for i in range(num_layers - 1):
		self.conv2 = GINConv(
			nn.SequentialLayer(
				nn.Dense(in_channels=hidden, n_units=hidden, act=tlx.ReLU),
				nn.Dense(in_channels=hidden, n_units=hidden, act=tlx.ReLU),
				nn.BatchNorm1d(num_features=hidden),
			), learn_eps=False)
		
		self.lin1 = nn.Dense(in_channels=hidden, n_units=hidden)
		self.lin2 = nn.Dense(in_channels=hidden, n_units=out_dim)
		self.dropout = tlx.nn.Dropout(keep=0.2)
	
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
		x = nn.ReLU()(self.lin1(x))
		x = self.dropout(x)
		x = self.lin2(x)
		return x

