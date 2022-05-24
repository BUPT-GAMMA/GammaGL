# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/22 23:59
# @Author  : clear
# @FileName: test.py.py

from subprocess import run

for i in range(5):
    run("CUDA_VISIBLE_DEVICES=1 python pgl_gcn.py >> log.txt", shell=True)
for i in range(5):
    run("CUDA_VISIBLE_DEVICES=1 python pgl_gat.py >> log.txt", shell=True)

