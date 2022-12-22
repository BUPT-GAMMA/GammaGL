import numpy as np
import tensorlayerx as tlx
from collections import Counter
from scipy.sparse import coo_matrix
from gammagl.utils import homophily, mask_to_index


class GEstimationN():
	"""
	Provide adjacency matrix estimation implementation based on the Expectation-Maximization(EM) algorithm.
	Parameters
	----------
	E: The actual observed number of edges between every pair of nodes (numpy.array)
	"""

	def __init__(self, data):
		self.num_class = data.num_classes
		graph = data[0]
		self.num_node = graph.num_nodes
		self.idx_train = tlx.convert_to_numpy(mask_to_index(graph.train_mask))
		self.label = tlx.convert_to_numpy(graph.y)
		row = graph.edge_index[0]
		col = graph.edge_index[1]
		data = np.ones(len(row))
		if tlx.BACKEND == 'torch':
			row = tlx.convert_to_numpy(row)
			col = tlx.convert_to_numpy(col)
		self.adj = coo_matrix((data, (row, col)), shape=(self.num_node, self.num_node)).toarray()

		self.output = None
		self.iterations = 0

		self.homophily = homophily(graph.edge_index, graph.y, method='node')

	def reset_obs(self):
		self.N = 0
		self.E = np.zeros((self.num_node, self.num_node), dtype=np.int64)

	def update_obs(self, output):
		self.E += output
		self.N += 1

	def revise_pred(self):
		# revise the prediction with train label
		# self.output = tlx.scatter_update(self.output, self.idx_train, tlx.gather(self.label, self.idx_train))
		self.output[self.idx_train] = self.label[self.idx_train]

	def E_step(self, Q):
		"""Run the Expectation(E) step of the EM algorithm.
		Parameters
		----------
		Q:
			The current estimation that each edge is actually present (numpy.array)

		Returns
		----------
		alpha:
			The estimation of true-positive rate (float)
		beta：
			The estimation of false-positive rate (float)
		O:
			The estimation of network model parameters (numpy.array)
		"""
		# Temporary variables to hold the numerators and denominators of alpha and beta
		an = Q * self.E
		an = np.triu(an, 1).sum()
		bn = (1 - Q) * self.E
		bn = np.triu(bn, 1).sum()
		ad = Q * self.N
		ad = np.triu(ad, 1).sum()
		bd = (1 - Q) * self.N
		bd = np.triu(bd, 1).sum()

		# Calculate alpha, beta
		alpha = an * 1. / (ad)
		beta = bn * 1. / (bd)

		O = np.zeros((self.num_class, self.num_class))

		n = []
		counter = Counter(self.output)
		for i in range(self.num_class):
			n.append(counter[i])

		a = self.output.repeat(self.num_node).reshape(self.num_node, -1)
		for j in range(self.num_class):
			c = (a == j)
			for i in range(j + 1):
				b = (a == i)
				O[i, j] = np.triu((b & c.T) * Q, 1).sum()
				if i == j:
					O[j, j] = 2. / (n[j] * (n[j] - 1)) * O[j, j]
				else:
					O[i, j] = 1. / (n[i] * n[j]) * O[i, j]
		return (alpha, beta, O)

	def M_step(self, alpha, beta, O):
		"""Run the Maximization(M) step of the EM algorithm.
		"""
		O += O.T - np.diag(O.diagonal())

		row = self.output.repeat(self.num_node)
		col = np.tile(self.output, self.num_node)
		tmp = O[row, col].reshape(self.num_node, -1)

		p1 = tmp * np.power(alpha, self.E) * np.power(1 - alpha, self.N - self.E)
		p2 = (1 - tmp) * np.power(beta, self.E) * np.power(1 - beta, self.N - self.E)
		Q = p1 * 1. / (p1 + p2 * 1.)
		return Q


	def EM(self, output, tolerance=.000001):
		"""Run the complete EM algorithm.
		Parameters
		----------
		tolerance:
			Determine the tolerance in the variantions of alpha, beta and O, which is acceptable to stop iterating (float)
		seed:
			seed for np.random.seed (int)
		Returns
		----------
		iterations:
			The number of iterations to achieve the tolerance on the parameters (int)
		"""
		# Record previous values to confirm convergence
		alpha_p = 0
		beta_p = 0

		self.output = tlx.convert_to_numpy(output)
		self.revise_pred()

		# Do an initial E-step with random alpha, beta and O
		# Beta must be smaller than alpha
		beta, alpha = np.sort(np.random.rand(2))
		O = np.triu(np.random.rand(self.num_class, self.num_class))

		# Calculate initial Q
		Q = self.M_step(alpha, beta, O)

		while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
			alpha_p = alpha
			beta_p = beta
			alpha, beta, O = self.E_step(Q)
			Q = self.M_step(alpha, beta, O)
			self.iterations += 1

		if self.homophily > 0.5:
			Q += self.adj
		return (alpha, beta, O, Q, self.iterations)
