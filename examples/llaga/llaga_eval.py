import random

import torch
import json
import argparse
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import os
root_path = "/home/yy3/graph-text-align/LLaGA"


def sbert(model_type, device):
    model = SentenceTransformer(model_type, device=device)
    return model

def get_sbert_embedding(model_type, texts, device):
    if model_type == 'sbert':
        model_type = 'all-MiniLM-L6-v2'
    sbert_model = sbert(model_type, f'cuda:{device}')
    sbert_embeds = sbert_model.encode(texts, batch_size=8, show_progress_bar=True)
    return torch.tensor(sbert_embeds)

def eval_arxiv_nd(res_path):
    data=torch.load(os.path.join(root_path,"dataset/ogbn-arxiv/processed_data.pt" ))
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    titles = data.title

    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            short_label = short_labels[y]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
            if short_label in ans:
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_arxiv_nc(res_path):
    data=torch.load(os.path.join(root_path,"dataset/ogbn-arxiv/processed_data.pt" ))
    labels=data.label_texts
    short_labels = [l[0:5] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    overall_correct=0
    strict_correct=0
    error=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if label.lower().strip() == ans.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower() in ans.lower() and sum([la.lower() in ans.lower() for la in short_labels])==1:
                overall_correct+=1
            else:
                error.append((ans, label))
            if args.sample > 0 and all_sample >= args.sample:
                break
    overall_acc = overall_correct/all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_lp(res_path):
    all_sample=0
    correct = 0
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            ans = res["text"].strip()
            label=res["gt"].strip()
            all_sample += 1
            if ("yes" in ans and "yes" in label) or ("yes" not in ans and "no" in label):
                correct += 1
            if args.sample > 0 and all_sample >=  args.sample:
                break
    acc = correct / all_sample
    print(f"Test samples: {all_sample}\ncorrect: {correct}\n acc: {acc:.4f}")

def eval_lprank(res_path):
    all_sample=0
    correct = 0
    y_true = []
    y_pred=[]
    with open(res_path, 'r') as f:
        for line in f:
            res = json.loads(line)
            logit = res["logit"]
            score = torch.softmax(torch.tensor(logit[:2]), dim=-1)[0].item()
            # score = logit[0]
            label=res["gt"].strip()
            if label == "yes":
                y_true.append(1)
            else:
                y_true.append(0)
            y_pred.append(score)
    auc = roc_auc_score(y_true, y_pred)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    acc = ((y_pred>0.5)==y_true).sum()/y_pred.shape[0]

    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    y_pos=y_pred[y_true==1]
    y_neg=y_pred[y_true==0]
    y_neg_sort, _ = torch.sort(y_neg)
    for n in [10,50,100,200,500,1000]:
        if n > y_neg_sort.shape[0]:
            break
        th = y_neg_sort[-n]
        h = (y_pos>th).sum()/y_pos.shape[0]
        print(f"Hits@{n}: {h:.4f}")

# here
def eval_products_nc(res_path):
    eval_set = set()
    data=torch.load(os.path.join(root_path,"dataset/ogbn-products/processed_data.pt" ))
    labels=data.label_names
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            if args.sample > 0 and all_sample >= args.sample:
                break
            all_sample+=1
            res = json.loads(line)
            if res['question_id'] in eval_set:
                print(f"{res['question_id']} repeat!!")
                return
            eval_set.add(res['question_id'])
            ans = res["text"].strip()
            y=ys[res["question_id"]][0]
            label=labels[y].strip()
            if label.lower()==ans.lower():
                strict_correct+=1
                overall_correct+=1
            elif label.lower() in ans.lower() and sum([l.lower() in ans.lower() for l in labels])<=2:
                overall_correct += 1

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")

def eval_products_nd(res_path):
    eval_set = set()
    data=torch.load(os.path.join(root_path,"dataset/ogbn-products/processed_data.pt" ))
    labels=data.label_names
    ys=data.y.numpy().tolist()

    all_sample=0
    all_correct=0
    gt = []
    out = []
    with open(res_path, 'r') as f:
        for line in f:
            if args.sample > 0 and all_sample >= args.sample:
                break
            all_sample+=1
            res = json.loads(line)
            if res['question_id'] in eval_set:
                print(f"{res['question_id']} repeat!!")
            eval_set.add(res['question_id'])
            ans = res["text"].strip()
            y=ys[res["question_id"]][0]
            label=labels[y].strip()
            if label.lower() in ans.lower():
                all_correct+=1
            desc = data.raw_texts[res['question_id']]
            assistant_prompt = f"This is an amazon product which can be categorized as {label}. It can be described as {desc}"
            gt.append(assistant_prompt)
            out.append(ans)
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}acc: {all_acc:.4f}")

    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding = F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding = F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim = (gt_embedding * out_embedding).sum(1).mean().item()
    gt_sim_matrix = torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n = gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)] = 0
    gt_sim = (gt_sim_matrix.sum() / (n * (n - 1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_pubmed_nc(res_path):
    data=torch.load(os.path.join(root_path,"dataset/pubmed/processed_data.pt" ))
    labels=data.label_texts
    short_labels = [l[18:] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    strict_correct=0
    overall_correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            short_label = short_labels[y]
            label=labels[y]
            if ans.lower().strip() == label.lower().strip():
                strict_correct+=1
                overall_correct+=1
            elif short_label.lower().strip() in ans.lower().strip() and sum([la.lower().strip() in ans.lower().strip() for la in short_labels]) == 1:
                overall_correct += 1
            if args.sample > 0 and all_sample >= args.sample:
                break

    overall_acc = overall_correct / all_sample
    strict_acc = strict_correct / all_sample
    print(f"Test samples: {all_sample}\nstrict_acc: {strict_acc:.4f}\noverall_acc: {overall_acc:.4f}")


def eval_pubmed_nd(res_path):
    data = torch.load(os.path.join(root_path,"dataset/pubmed/processed_data.pt" ))
    labels = data.label_texts
    short_labels = [l[18:] for l in labels]
    ys = data.y.numpy().tolist()

    titles = data.title
    abs = data.abs

    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            short_label = short_labels[y]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
            if short_label in ans:
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")


def eval_cora_nc(res_path):
    data=torch.load(os.path.join(root_path,"dataset/cora/processed_data.pt" ))
    labels=data.label_texts
    short_labels = [l.split('_')[0] for l in labels]
    ys=data.y.numpy().tolist()

    all_sample=0
    correct=0
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            y=ys[res["question_id"]]
            label=labels[y]
            short_label=short_labels[y]
            if short_label.strip().lower() in ans.strip().lower() and sum([l.strip().lower() in ans.strip().lower() for l in short_labels])==1:
                correct+=1
    acc=correct/all_sample
    print(f"Test samples: {all_sample}\nacc: {acc:.4f}")



def eval_cora_nd(res_path):
    data = torch.load(os.path.join(root_path,"dataset/cora/processed_data.pt" ))
    labels = data.label_texts
    ys = data.y.numpy().tolist()

    titles = data.title
    all_sample=0
    short_correct=0
    all_correct=0
    gt=[]
    out=[]
    with open(res_path, 'r') as f:
        for line in f:
            all_sample+=1
            res = json.loads(line)
            ans = res["text"]
            id=res["question_id"]
            y=ys[id]
            label=labels[y]
            if label.strip() in ans.strip():
                all_correct+=1
                short_correct+=1
            out.append(ans)
            gt.append(f"This is a paper in {label} domain, it's about {titles[id]}.")
    short_acc = short_correct/all_sample
    all_acc = all_correct / all_sample
    print(f"Test samples: {all_sample}\nshort_correct: {short_correct}\nshort_acc: {short_acc:.4f}\nall_correct: {all_correct}\nall_acc: {all_acc:.4f}")
    gt_embedding = get_sbert_embedding("sbert", gt, 0)
    out_embedding = get_sbert_embedding("sbert", out, 0)
    gt_embedding=F.normalize(gt_embedding, p=2, eps=1e-6, dim=1)
    out_embedding=F.normalize(out_embedding, p=2, eps=1e-6, dim=1)
    predict_sim=(gt_embedding*out_embedding).sum(1).mean().item()
    gt_sim_matrix=torch.mm(gt_embedding, gt_embedding.transpose(0, 1)).detach().cpu()
    n=gt_sim_matrix.shape[0]
    gt_sim_matrix[torch.eye(n, dtype=torch.bool)]=0
    gt_sim=(gt_sim_matrix.sum()/(n*(n-1))).item()
    print(f"Predict similarity {predict_sim: .4f}, Pairwise similarity: {gt_sim: .4f}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="./results/llaga-opt-2.7b-v1-simteg_all_origin_tape_multihop-laplacian_-1-2-10-linear-only-train-pretrain_acc1_nc_test_nc.jsonl")
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    func_dict = {
        "arxiv":{
            "nc": eval_arxiv_nc,
            "nd": eval_arxiv_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "products": {
            "nc": eval_products_nc,
            "nd": eval_products_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "pubmed": {
            "nc": eval_pubmed_nc,
            "nd": eval_pubmed_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
        "cora": {
            "nc": eval_cora_nc,
            "nd": eval_cora_nd,
            "lp": eval_lp,
            "lprank": eval_lprank
        },
    }

    func=func_dict[args.dataset][args.task]
    func(args.res_path)