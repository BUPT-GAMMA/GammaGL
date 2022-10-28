import os
import tqdm
import pickle
import collections
import multiprocessing

from gammagl.models import Generator, Discriminator


class GraphGAN(object):
    r"""
        GraphGAN Model proposed in `"GraphGAN: Graph Representation Learning with Generative Adversarial Nets"
        <https://arxiv.org/pdf/1711.08267.pdf>`_ paper

        .. math:: \min_{\Theta _{G}}\max_{\Theta _{D}}V(G,D)=\sum_{c=1}^{V}(E_{v\sim p_{true}(\cdot | v_{c})}[logD(v,v_{c};\Theta _{D})]+E_{v\sim G(\cdot |v_{c};\Theta _{G})}[log(1-D(v,v_{c};\Theta _{D}))])

        The optimal parameters of the generator and the discriminator can be learned by alternately maximizing and
        minimizing the value function :math:`V (G, D)`. In each iteration, discriminator D is trained with positive
        samples from :math:`p_{true}(\cdot |v_{c})` and negative samples from generator :math:`G(\cdot |v_{c};\Theta
        _{G})`, and generator G is updated with policy gradient under the guidance of :math:`D`. Competition between :math:`G` and :math:`D`
        drives both of them to improve their methods until :math:`G` is indistinguishable from the true connectivity
        distribution.

        Parameters
        ----------
            args: Namespace
                Parameters setting

    """

    def __init__(self, n_node, graph, node_embed_init_d, node_embed_init_g, cache_dir, multi_processing):
        self.n_node = n_node
        self.graph = graph
        self.root_nodes = [i for i in range(self.n_node)]

        self.node_embed_init_d = node_embed_init_d
        self.node_embed_init_g = node_embed_init_g

        self.trees = None
        # construct BFS-trees
        if os.path.isfile(f'{cache_dir}/CA-GrQc.pkl'):
            print("reading BFS-trees from cache...")
            pickle_file = open(f'{cache_dir}/CA-GrQc.pkl', 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            
            pickle_file = open(f'{cache_dir}/CA-GrQc.pkl', 'wb')
            if multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

        self.discriminator = Discriminator(self.n_node, self.node_embed_init_d)
        self.generator = Generator(self.n_node, self.node_embed_init_g)

    def construct_trees_with_mp(self, nodes):
        """
        Use the multiprocessing to speed up trees construction

        Parameters
        ----------
            nodes : list
                List of nodes in the graph

        """

        cores = multiprocessing.cpu_count() // 2
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """
        Use BFS algorithm to construct the BFS-trees

        Parameters
        ----------
            nodes : list
                List of nodes in the graph

        Returns
        -------
            dict
                root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]

        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees
