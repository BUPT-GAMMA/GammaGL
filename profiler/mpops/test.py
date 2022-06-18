# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/05/16 18:31
# @Author  : clear
# @FileName: test.py.py

from subprocess import run

for i in range(5):
    run('python pyg_gpu.py', shell=True)

for i in range(5):
    run('python pd_gpu.py', shell=True)
for i in range(5):
    run('python pd_gpu.py', shell=True)
for i in range(5):
    run('python ms_gpu.py', shell=True)
for i in range(5):
    run('python th_gpu.py', shell=True)
