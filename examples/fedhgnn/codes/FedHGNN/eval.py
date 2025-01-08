import os
from tensorlayerx.backend.ops.torch_backend import topk
os.environ['TL_BACKEND'] = 'torch'
import numpy as np
import math

def getP(ranklist, gtItems):
    p = 0
    for item in ranklist:
        if item in gtItems:
            p += 1
    return p * 1.0 / len(ranklist)

def getR(ranklist, gtItems):
    r = 0
    for item in ranklist:
        if item in gtItems:
            r += 1
    return r * 1.0 / len(gtItems)


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getDCG(ranklist, gtItems):
    dcg = 0.0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item in gtItems:
            dcg += 1.0 / math.log(i + 2)
    return  dcg

def getIDCG(ranklist, gtItems):
    idcg = 0.0
    i = 0
    for item in ranklist:
        if item in gtItems:
            idcg += 1.0 / math.log(i + 2)
            i += 1
    return idcg

def getNDCG(ranklist, gtItems):
    dcg = getDCG(ranklist, gtItems)
    idcg = getIDCG(ranklist, gtItems)
    if idcg == 0:
        return 0
    return dcg / idcg


'''下面是两个大指标(recall或ndcg)，4个小指标的计算代码'''
#指标1 top_k=5 或 10   得到recall@5 或recall@10 指标
def evaluate_recall(rating, ground_truth, top_k):
    _, rating_k = topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()

    hit = 0 #
    for i, v in enumerate(rating_k):
        if v in ground_truth:
            hit += 1

    recall = hit / len(ground_truth)
    return recall

#指标2 top_k = 5 或 10 得到ndcg@5 或ndcg@10 指标
def evaluate_ndcg(rating, ground_truth, top_k):#参照NDCG的定义
    _, rating_k = topk(rating, top_k)#values, indices
    rating_k = rating_k.cpu().tolist() #indices
    dcg, idcg = 0., 0.

    for i, v in enumerate(rating_k):
        if i < len(ground_truth):#前len（）个是真实交互的
            idcg += (1 / np.log2(2 + i))#这里相关性为0或1（真实交互为1，未交互为0）
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    ndcg = dcg / idcg
    return ndcg
