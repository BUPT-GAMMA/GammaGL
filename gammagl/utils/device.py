# !/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2022/06/21 13:33
# @Author  : clear
# @FileName: device.py

import tensorlayerx as tlx

def set_device(id=0):
    if id >= 0:
        # if cuda is available
        tlx.set_device("GPU", id)
    else:
        tlx.set_device("CPU", id)

