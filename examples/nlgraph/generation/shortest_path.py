import os
from random import randint, shuffle, random
import networkx as nx
import argparse
from tqdm import tqdm

class Generator:
    def __init__(self,num_of_nodes = 10, edge_probability = 0.35, max_weight = 4):
        self.num_of_nodes = num_of_nodes
        self.edge_probability = edge_probability
        self.max_weight = max_weight

    def generate_graph(self):
        l = randint(2, 6)
        while True:
            idx = list(range(self.num_of_nodes))
            shuffle(idx)
            G = nx.Graph()
            G.add_nodes_from(range(self.num_of_nodes))
            for u in list(G.nodes()):
                for v in list(G.nodes()):
                    if u < v and random() < self.edge_probability:
                        weight = randint(1,self.max_weight)
                        G.add_edge(idx[u], idx[v], weight = weight)
            if nx.is_connected(G):
                q = []
                shuffle(idx)
                for u in list(G.nodes()):
                    if len(q) > 0:
                        break
                    for v in list(G.nodes()):
                        if u != v and not G.has_edge(idx[u], idx[v]) and nx.shortest_path_length(G, source=idx[u], target=idx[v])>=l:
                            q = [idx[u],idx[v]]
                            break
                if len(q) > 0:
                    return G, q
    def generate(self):
        G, q = self.generate_graph()
        return G, q

parser = argparse.ArgumentParser(description="shortest path generation")
parser.add_argument('--mode', type=str, default="easy", help='mode (default: easy)')
args = parser.parse_args()    
assert args.mode in ["easy","hard"]
p_list = [0.5, 0.7, 0.9]
standard_num = 10
if args.mode == "easy": # 6*10*3
    n_min = 5
    n_max = 10
    g_num = 10 * 2
    max_weight = 4
elif args.mode =="hard": # 10*10*2
    n_min = 11
    n_max = 20
    p_list = [0.2, 0.25]
    g_num = 10 * 6
    max_weight = 10

newpath = r'D:\python\LLM\shortest_path\graph'+ '\\'+args.mode
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(newpath + '\\' + "full")
    os.makedirs(newpath + '\\' + "standard") 
graph_index, standard_index = 0, 0
for num in tqdm(range(n_min, n_max+1)):
    for edge_probability in p_list:
        for generate_num in range(g_num):
            generator = Generator(num_of_nodes=num,edge_probability=edge_probability, max_weight=max_weight)
            Graph, q = generator.generate()
            edge = list(Graph.edges())
            with open("./graph/"+args.mode+"/full/graph"+str(graph_index)+".txt","w") as f:
                f.write(str(Graph.number_of_nodes())+' '+str(Graph.number_of_edges())+'\n')
                for i in range(len(Graph.edges())):
                    if random() < 0.5:
                        f.write(str(edge[i][0])+' '+str(edge[i][1])+' '+str(Graph[edge[i][0]][edge[i][1]]["weight"])+'\n')
                    else:
                        f.write(str(edge[i][1])+' '+str(edge[i][0])+' '+str(Graph[edge[i][0]][edge[i][1]]["weight"])+'\n')
                f.write(str(q[0])+' '+str(q[1])+'\n')
            if generate_num < standard_num:
                with open("./graph/"+args.mode+"/standard/graph"+str(standard_index)+".txt","w") as f:
                    f.write(str(Graph.number_of_nodes())+' '+str(Graph.number_of_edges())+'\n')
                    for i in range(len(Graph.edges())):
                        if random() < 0.5:
                            f.write(str(edge[i][0])+' '+str(edge[i][1])+' '+str(Graph[edge[i][0]][edge[i][1]]["weight"])+'\n')
                        else:
                            f.write(str(edge[i][1])+' '+str(edge[i][0])+' '+str(Graph[edge[i][0]][edge[i][1]]["weight"])+'\n')
                    f.write(str(q[0])+' '+str(q[1])+'\n')
                standard_index += 1
            graph_index += 1
