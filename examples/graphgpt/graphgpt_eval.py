import json
import os.path as osp
import os
import torch as th
import re
import pandas as pd
from tqdm import tqdm 
from sklearn.metrics import classification_report

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='arxiv')
args = parser.parse_args()

label_to_idx = {
    "cora":{"databases, object oriented": 29, "operating systems, memory management": 59, "data structures  algorithms and theory, quantum computing": 24, "artificial intelligence, planning": 13, "artificial intelligence, knowledge representation": 4, "artificial intelligence, data mining": 1, "artificial intelligence, vision and pattern recognition": 17, "artificial intelligence, machine learning, case-based": 5, "artificial intelligence, agents": 0, "artificial intelligence, machine learning, probabilistic methods": 8, "encryption and compression, security": 36, "operating systems, distributed": 57, "human computer interaction, interface design": 46, "artificial intelligence, machine learning, genetic algorithms": 6, "human computer interaction, graphics and virtual reality": 45, "artificial intelligence, machine learning, rule learning": 10, "programming, functional": 63, "programming, object oriented": 67, "encryption and compression, encryption": 35, "databases, performance": 30, "networking, protocols": 54, "data structures  algorithms and theory, randomized": 25, "data structures  algorithms and theory, formal languages": 20, "data structures  algorithms and theory, parallel": 23, "programming, software development": 69, "programming, compiler design": 61, "artificial intelligence, machine learning, theory": 11, "artificial intelligence, machine learning, neural networks": 7, "programming, logic": 66, "databases, relational": 32, "information retrieval, retrieval": 52, "programming, debugging": 62, "networking, wireless": 56, "artificial intelligence, theorem proving": 16, "databases, temporal": 33, "encryption and compression, compression": 34, "information retrieval, filtering": 51, "data structures  algorithms and theory, computational complexity": 18, "programming, garbage collection": 64, "artificial intelligence, machine learning, reinforcement learning": 9, "human computer interaction, multimedia": 47, "hardware and architecture, vlsi": 43, "artificial intelligence, nlp": 12, "hardware and architecture, microprogramming": 42, "operating systems, fault tolerance": 58, "programming, java": 65, "operating systems, realtime": 60, "human computer interaction, cooperative": 44, "artificial intelligence, speech": 15, "databases, deductive": 28, "artificial intelligence, robotics": 14, "data structures  algorithms and theory, logic": 22, "networking, routing": 55, "hardware and architecture, logic design": 40, "hardware and architecture, distributed architectures": 37, "data structures  algorithms and theory, hashing": 21, "programming, semantics": 68, "artificial intelligence, games and search": 3, "databases, concurrency": 27, "data structures  algorithms and theory, sorting": 26, "human computer interaction, wearable computers": 48, "information retrieval, digital library": 49, "artificial intelligence, expert systems": 2, "information retrieval, extraction": 50, "data structures  algorithms and theory, computational geometry": 19, "databases, query evaluation": 31, "networking, internet": 53, "hardware and architecture, memory structures": 41, "hardware and architecture, high performance computing": 38, "hardware and architecture, input output and storage": 39},
    "pubmed":{"Experimentally induced diabetes": 0, "Type 2 diabetes": 2, "Type 1 diabetes": 1}
}



data_list = []
folder = 'output_stage_2_{}_nc'.format(args.dataset)
for filename in os.listdir(folder):
    if filename.endswith('.json'): 
        file_path = os.path.join(folder, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            data_list.extend(data)

print(data_list[1])

graph_data = th.load('/local/yy3/graphgpt/data/graph_data_all.pt')[args.dataset]
labels = graph_data.y

def cal_map(): 
    label_dict = {}
    if args.dataset == "arxiv":
        df = pd.read_csv(os.path.expanduser('~/datasets/OGB/ogbn_arxiv/mapping/labelidx2arxivcategeory.csv.gz'), compression='gzip')
        for index, line in df.iterrows(): 
            lb = line['arxiv category'].split(' ')[-1]
            lb_new = 'cs.' + lb.upper()
            label_dict[lb_new] = line['label idx']
    else:
        label_dict = label_to_idx[args.dataset]
    return label_dict

class_map = cal_map()

inverse_class_map = {}
for lb, lb_id in class_map.items():
    inverse_class_map[lb_id] = lb
    

pattern = r"cs\.[A-Z]{2}"


topk = 3

correct = 0
total = len(data_list)

trues = []
preds = []

for instruct_item in tqdm(data_list): 
    nid = instruct_item['node_idx']
    gpt_res = instruct_item['res']


    true_y = labels[nid]

    pred_y = []
    if args.dataset == "arxiv":
        matches = list(set(re.findall(pattern, gpt_res))) # pred
        sorted_matches = sorted(matches, key=lambda x: gpt_res.index(x))
        for m in sorted_matches:
            try:
                pred_y.append(class_map[m])
            except: 
                pass
        try:
            # print(sorted_matches)
            preds.append(pred_y[0])
        except:   
            preds.append(-1)
    else:
        for lb, lb_id in class_map.items():
            if lb in gpt_res:
                pred_y.append(lb_id)
        try:
            # print(sorted_matches)
            preds.append(pred_y[0])
        except:   
            preds.append(-1)
    trues.append(true_y.item())
    res_tmp = 1 if true_y in pred_y[:topk] else 0
    correct = correct + 1 if true_y in pred_y[:topk] else correct

acc = correct / total

print("Accuracy:", acc)

report = classification_report(trues, preds, digits=6)

print(report)