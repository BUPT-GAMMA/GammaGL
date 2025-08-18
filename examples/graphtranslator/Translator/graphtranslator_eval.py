import numpy as np
import argparse
import random
import sys
import os
import re
import pandas as pd
os.environ['TL_BACKEND'] = "torch"

import logging
import tensorlayerx as tlx
from datetime import datetime

from utils import Config, setup_logger

from runner_base import RunnerBase
from arxiv_text_pair_datasets import ArxivTextPairDataset
from gammagl.models import TranslatorCHATGLMArxiv



def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--distributed", action='store_const', default=True, const=True)
    parser.add_argument('--random_seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--cfg-path", default="./pretrain_arxiv_generate_stage2.yaml", help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed

    random.seed(seed)
    np.random.seed(seed)
    tlx.set_seed(seed)  # Set the seed for tensorlayerx

    if os.environ['TL_BACKEND'] == "torch":
        import torch.backends.cudnn as cudnn
        import torch
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = True

def build_datasets(cfg):
    datasets = dict()

    datasets_config = cfg.datasets_cfg

    assert len(datasets_config) > 0, "At least one dataset has to be specified."

    for name in datasets_config:
        dataset_config = datasets_config[name]
        logging.info("Building datasets...")

        dataset = dict()
        dataset["train"] = ArxivTextPairDataset(
            cfg=dataset_config,
            mode='train'
        )

        datasets[name] = dataset

    return datasets

def get_topk_predictions(node_dict, k):
    topk_predictions = {}
    for node, probabilities in node_dict.items():
        topk_predictions[node] = probabilities[:k]
    return topk_predictions


def legality_rate(node2pred):
    patterns = ["cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY", "Artificial Intelligence", "Hardware Architecture", "Computational Complexity", "Computational Engineering, Finance, and Science", "Computational Geometry", "Computation and Language", "Cryptography and Security", "Computer Vision and Pattern Recognition", "Computers and Society", "Databases", "Distributed, Parallel, and Cluster Computing", "Digital Libraries", "Discrete Mathematics", "Data Structures and Algorithms", "Emerging Technologies", "Formal Languages and Automata Theory", "General Literature", "Graphics", "Computer Science and Game Theory", "Human-Computer Interaction", "Information Retrieval", "Information Theory", "Machine Learning", "Logic in Computer Science", "Multiagent Systems", "Multimedia", "Mathematical Software", "Numerical Analysis", "Neural and Evolutionary Computing", "Networking and Internet Architecture", "Other Computer Science", "Operating Systems", "Performance", "Programming Languages", "Robotics", "Symbolic Computation", "Sound", "Software Engineering", "Social and Information Networks", "Systems and Control","Computer Vision","Pattern Recognition"]
    label_map = {'Numerical Analysis': 0,'Multimedia': 1,'Logic in Computer Science': 2,'Computers and Society': 3,'Cryptography and Security': 4,'Distributed, Parallel, and Cluster Computing': 5,'Human-Computer Interaction': 6,'Computational Engineering, Finance, and Science': 7,'Networking and Internet Architecture': 8,'Computational Complexity': 9,'Artificial Intelligence': 10,'Multiagent Systems': 11,'General Literature': 12,'Neural and Evolutionary Computing': 13,'Symbolic Computation': 14,'Hardware Architecture': 15,'Computer Vision and Pattern Recognition': 16,'Pattern Recognition': 16,'Computer Vision': 16,'Graphics': 17,'Emerging Technologies': 18,'Systems and Control': 19,'Computational Geometry': 20,'Other Computer Science': 21,'Programming Languages': 22,'Software Engineering': 23,'Machine Learning': 24,'Sound': 25,'Social and Information Networks': 26,'Robotics': 27,'Information Theory': 28,'Performance': 29,'Computation and Language': 30,'Information Retrieval': 31,'Mathematical Software': 32,'Formal Languages and Automata Theory': 33,'Data Structures and Algorithms': 34,'Operating Systems': 35,'Computer Science and Game Theory': 36,'Databases': 37,'Digital Libraries': 38,'Discrete Mathematics': 39,'cs.NA': 0,'cs.MM': 1,'cs.LO': 2,'cs.CY': 3,'cs.CR': 4,'cs.DC': 5,'cs.HC': 6,'cs.CE': 7,'cs.NI': 8,'cs.CC': 9,'cs.AI': 10,'cs.MA': 11,'cs.GL': 12,'cs.NE': 13,'cs.SC': 14,'cs.AR': 15,'cs.CV': 16,'cs.GR': 17,'cs.ET': 18,'cs.SY': 19,'cs.CG': 20,'cs.OH': 21,'cs.PL': 22,'cs.SE': 23,'cs.LG': 24,'cs.SD': 25,'cs.SI': 26,'cs.RO': 27,'cs.IT': 28,'cs.PF': 29,'cs.CL': 30,'cs.IR': 31,'cs.MS': 32,'cs.FL': 33,'cs.DS': 34,'cs.OS': 35,'cs.GT': 36,'cs.DB': 37,'cs.DL': 38,'cs.DM': 39}
    print("Total class number:", len(patterns))
    assert len(patterns) == len(label_map), "patterns and label_map should have the same size"

    count = 0
    node2digitallabel = {}
    for node, pred_list in node2pred.items():
        matches = []
        for pred in pred_list:
            for pattern in patterns:
                match_label = re.findall(pattern, pred)
                if len(match_label) > 2:
                    match_label = list(set(match_label))
                    label1 = label_map[match_label[0]]
                    matches.append(label1)
                elif len(match_label) == 2:
                    label1 = label_map[match_label[0]]
                    label2 = label_map[match_label[1]]
                    if label1 != label2:
                        print("error")
                    else:
                        matches.append(label1)
                elif len(match_label) == 1:
                    label1 = label_map[match_label[0]]
                    matches.append(label1)
        matches = list(set(matches))
        node2digitallabel[int(node)] = list(set(matches))
        if len(matches)>0:
            count += 1

    print(f"Total sample number: {count}")
    print(f"Legality rate: {round(100*count/len(node2pred),2) if len(node2pred) > 0 else 0.0}%")

    return node2digitallabel, count


def read_data(label_file, pred_file):
    df_node2label = pd.read_csv(label_file)
    node2label = dict(zip(df_node2label['node_id'], df_node2label['digital_label']))
    df_pred = pd.read_csv(pred_file, sep='\t', names=['node', 'summary', 'pred'])
    node2pred = {}
    for _, row in df_pred.iterrows():
        node = int(row.iloc[0])
        node2pred[node] = row.iloc[2].split("\n")
    return node2label, node2pred


def evaluation(label_file, pred_file):
    node2label, node2pred = read_data(label_file, pred_file)
    node2digitallabel, count = legality_rate(node2pred)

    for k in [1,3,5]:
        acc_count = 0
        node2digitallabel_k = get_topk_predictions(node2digitallabel, k )
        for node, pred_list in node2digitallabel_k.items():
            label = node2label[node]
            if len(pred_list) > 0 and label in pred_list:
                acc_count += 1

        print(f"Top@{k} Accuracy: {round(100*acc_count/count,2) if count > 0 else 0.0}%")

def main(job_id):
    cfg = Config(parse_args())

    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()

    # generate
    datasets = build_datasets(cfg)
    model = TranslatorCHATGLMArxiv.from_config(cfg.model_cfg)

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, model=model, datasets=datasets
    )
    runner.translator_generate()

    # evaluate
    label_file = "../data/arxiv_test.csv"
    pred_file = "../data/pred.txt"
    evaluation(label_file, pred_file)


if __name__ == "__main__":
    job_id = datetime.now().strftime("%Y%m%d%H%M")[:-1]

    main(job_id)
