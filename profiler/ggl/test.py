# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/22 23:59
# @Author  : clear
# @FileName: test.py.py

from subprocess import run

for i in range(5):
    run("TL_BACKEND=torch CUDA_VISIBLE_DEVICES=1 python gcn_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=mindspore CUDA_VISIBLE_DEVICES=1 python gcn_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=1 python gcn_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=paddle CUDA_VISIBLE_DEVICES=1 python gcn_trainer.py >> log.txt", shell=True)

for i in range(5):
    run("TL_BACKEND=torch CUDA_VISIBLE_DEVICES=1 python gat_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=mindspore CUDA_VISIBLE_DEVICES=1 python gat_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=1 python gat_trainer.py >> log.txt", shell=True)
for i in range(5):
    run("TL_BACKEND=paddle CUDA_VISIBLE_DEVICES=1 python gat_trainer.py >> log.txt", shell=True)
