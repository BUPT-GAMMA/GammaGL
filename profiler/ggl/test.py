# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/22 23:59
# @Author  : clear
# @FileName: test.py.py

from subprocess import run
import os

logfile = 'log.txt'
gpu = 8
iter = 5
dataset = 'pubmed'

if os.path.exists('log.txt'):
    os.remove('log.txt')

# GCN
for i in range(iter):
    run("TL_BACKEND=torch CUDA_VISIBLE_DEVICES={} python gcn_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=mindspore CUDA_VISIBLE_DEVICES={} python gcn_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=tensorflow CUDA_VISIBLE_DEVICES={} python gcn_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=paddle CUDA_VISIBLE_DEVICES={} python gcn_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)

# GAT
for i in range(iter):
    run("TL_BACKEND=torch CUDA_VISIBLE_DEVICES={} python gat_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=mindspore CUDA_VISIBLE_DEVICES={} python gat_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=tensorflow CUDA_VISIBLE_DEVICES={} python gat_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)
for i in range(iter):
    run("TL_BACKEND=paddle CUDA_VISIBLE_DEVICES={} python gat_trainer.py --dataset {} >> {}"
        .format(gpu, dataset, logfile), shell=True)

aa = []
bb = []
cc = []
with open(logfile) as f:
    lines = f.readlines()
    counter = 0
    for line in lines:
        a, b, c = line.split()
        aa.append(float(a))
        bb.append(float(b))
        cc.append(float(c))
        counter += 1
        if counter % iter == 0:
            print(sum(aa)/iter, sum(bb)/iter, sum(cc)/iter)
            aa = []
            bb = []
            cc = []
