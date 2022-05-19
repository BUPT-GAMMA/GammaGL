import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.layers.conv.gin_conv import GINConv
from gammagl.layers.pool.glob import global_mean_pool, global_max_pool, global_sum_pool


#
# class GINModel(nn.Module):
# 	def __init__(self, name=None, input_dim=None, hidden=None, out_dim=None):
#
# 		super().__init__(name=name)
# 		self.conv1 = GINConv(
# 			nn.Sequential([
# 				nn.Linear(in_features=input_dim, out_features=hidden, act=nn.activation.ReLU,
# 						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
# 				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU,
# 						 b_init=tlx.initializers.random_uniform(minval=-1, maxval=1)),
# 				nn.BatchNorm1d(num_features=hidden)]
# 			), learn_eps=False)
# 		# for i in range(num_layers - 1):
# 		self.conv2 = GINConv(
# 			nn.Sequential([
# 				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU),
# 				nn.Linear(in_features=hidden, out_features=hidden, act=nn.activation.ReLU),
# 				nn.BatchNorm1d(num_features=hidden)]
# 			), learn_eps=False)
#
# 		self.lin1 = nn.Linear(in_features=hidden, out_features=hidden)
# 		self.lin2 = nn.Linear(in_features=hidden, out_features=out_dim)
# 		self.act = nn.ReLU()
# 		self.dropout = tlx.nn.Dropout(p=0.2)
#
# 	def reset_parameters(self):
# 		self.conv1.reset_parameters()
# 		for conv in self.convs:
# 			conv.reset_parameters()
# 		self.lin1.reset_parameters()
# 		self.lin2.reset_parameters()
#
# 	def forward(self, x, edge_index, batch):
# 		x = self.conv1(x, edge_index)
# 		x = self.conv2(x, edge_index)
# 		x = global_mean_pool(x, batch)
# 		x = self.act(self.lin1(x))
# 		x = self.dropout(x)
# 		x = self.lin2(x)
# 		return x


class ApplyNodeFunc(nn.Module):
	"""Update the node feature hv with MLP, BN and ReLU."""

	def __init__(self, mlp):
		super(ApplyNodeFunc, self).__init__()
		self.mlp = mlp
		self.bn = nn.BatchNorm1d(num_features=self.mlp.output_dim)
		self.act = nn.ReLU()

	def forward(self, h):
		h = self.mlp(h)
		h = self.bn(h)
		h = self.act(h)
		return h


class MLP(nn.Module):
	"""MLP with linear output"""

	def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
		"""MLP layers construction

        Parameters
        ----------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
		super(MLP, self).__init__()
		self.linear_or_not = True  # default is linear model
		self.num_layers = num_layers
		self.output_dim = output_dim

		if num_layers < 1:
			raise ValueError("number of layers should be positive!")
		elif num_layers == 1:
			# Linear model
			self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
		else:
			# Multi-layer model
			self.linear_or_not = False
			self.linears = nn.Sequential()
			self.batch_norms = nn.Sequential()

			self.linears.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
			for layer in range(num_layers - 2):
				self.linears.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
			self.linears.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))

			for layer in range(num_layers - 1):
				self.batch_norms.append(nn.BatchNorm1d(num_features=hidden_dim))
		self.act = nn.ReLU()

	def forward(self, x):
		if self.linear_or_not:
			# If linear model
			return self.linear(x)
		else:
			# If MLP
			h = x
			for i in range(self.num_layers - 1):
				h = self.act(self.batch_norms[i](self.linears[i](h)))
			return self.linears[-1](h)


class GINModel(nn.Module):
	"""GIN model

	model parameters setting

	Parameters
	----------
	num_layers: int
		The number of linear layers in the neural network
	num_mlp_layers: int
		The number of linear layers in mlps
	input_dim: int
		The dimensionality of input features
	hidden_dim: int
		The dimensionality of hidden units at ALL layers
	output_dim: int
		The number of classes for prediction
	final_dropout: float
		dropout ratio on the final linear layer
	learn_eps: boolean
		If True, learn epsilon to distinguish center nodes from neighbors
		If False, aggregate neighbors and center nodes altogether.
	neighbor_pooling_type: str
		how to aggregate neighbors (sum, mean, or max)
	graph_pooling_type: str
		how to aggregate entire nodes in a graph (sum, mean or max)
	"""

	def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
	             output_dim, final_dropout, learn_eps, graph_pooling_type,
	             neighbor_pooling_type,*args, **kwargs):
		super(GINModel, self).__init__(*args, **kwargs)
		self.num_layers = num_layers
		self.learn_eps = learn_eps

		# List of MLPs
		self.ginlayers = nn.Sequential()
		self.batch_norms = nn.Sequential()

		for layer in range(self.num_layers - 1):
			if layer == 0:
				mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
			else:
				mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

			self.ginlayers.append(
				GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
			self.batch_norms.append(nn.BatchNorm1d(num_features=hidden_dim))

		# Linear function for graph poolings of output of each layer
		# which maps the output of different layers into a prediction score
		self.linears_prediction = nn.Sequential()

		for layer in range(num_layers):
			if layer == 0:
				self.linears_prediction.append(
					nn.Linear(in_features=input_dim, out_features=output_dim))
			else:
				self.linears_prediction.append(
					nn.Linear(in_features=hidden_dim, out_features=output_dim))

		self.drop = nn.Dropout(final_dropout)
		self.act = nn.ReLU()
		if graph_pooling_type == 'sum':
			self.pool = global_sum_pool
		elif graph_pooling_type == 'mean':
			self.pool = global_mean_pool
		elif graph_pooling_type == 'max':
			self.pool = global_max_pool
		else:
			raise NotImplementedError

	def forward(self, x, edge_index, batch):
		hidden_rep = [x]
		for i in range(self.num_layers - 1):
			x = self.ginlayers[i](x, edge_index)
			x = self.batch_norms[i](x)
			x = self.act(x)
			hidden_rep.append(x)
		# perform pooling over all nodes in each graph in every layer
		score_over_layer = 0

		for i, x in enumerate(hidden_rep):
			pooled_h = self.pool(x, batch)
			score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
		return score_over_layer

