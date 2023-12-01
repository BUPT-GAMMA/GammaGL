# -*- coding: utf-8 -*-
# @author WuJing
# @created 2023/4/9

import tensorlayerx as tlx
import json
import os
import os.path as osp
from typing import Dict, Union
import hashlib
_home_path = osp.expanduser("~")
_ggl_dirname = ".ggl"
ggl_path = osp.join(_home_path, _ggl_dirname)
dataset_root = ""
config_default_dict: Union[Dict, None] = None
config_dict: Union[Dict, None] = None


def init_config():
    global config_default_dict
    global config_dict
    if config_dict is not None and config_default_dict is not None:
        return
    if osp.exists(ggl_path):
        ggl_config_default_path = "ggl_config_default.json"
        # ggl_config_default_path = osp.join(os.getcwd(), ggl_config_default_path)
        ggl_config_default_path = osp.join(osp.dirname(osp.abspath(__file__)), ggl_config_default_path)
        ggl_config_file = "ggl_config.json"
        ggl_config_path = osp.join(ggl_path, ggl_config_file)
        if osp.exists(ggl_config_path):
            with open(ggl_config_path) as f:
                config_dict = json.load(f)
            with open(ggl_config_default_path) as f:
                config_default_dict = json.load(f)
        else:
            from shutil import copyfile
            copyfile(ggl_config_default_path, ggl_config_path)
            with open(ggl_config_default_path) as f:
                config_default_dict = json.load(f)
            config_dict = config_default_dict

    else:
        try:
            os.mkdir(ggl_path)
            init_config()
        except:
            raise FileExistsError("cannot init ~/.ggl dir!")


def read_config():
    global config_dict
    global config_default_dict
    global dataset_root

    tlx.BACKEND = config_dict.get("tlx_backend") or config_default_dict.get("tlx_backend")
    tlx.BACKEND = os.getenv("TL_BACKEND", tlx.BACKEND)

    dataset_path = config_dict.get("dataset_root") or config_default_dict.get("dataset_root")
    # '@' stands for ~/.ggl, remove it to use abs path
    if dataset_path.startswith("@"):
        dataset_root = osp.join(ggl_path, dataset_path[1:])
    else:
        dataset_root = dataset_path
    if not osp.exists(dataset_root):
        try:
            os.mkdir(dataset_root)
        except:
            raise FileExistsError(f"can not make dataset dir:{dataset_root}")

    config_interpreter_str = f'''{'=' * 100}
Gammagl Global Config Info:
    TLX_BACKEND: {tlx.BACKEND}
    DATASET_ROOT: {dataset_root}
{'=' * 100}'''
    print(config_interpreter_str)


def global_config_init():
    init_config()
    read_config()


def get_dataset_root():
    global dataset_root
    return dataset_root


def get_dataset_meta_path():
    global ggl_path
    dataset_meta_path = osp.join(ggl_path, 'dataset_meta.json')
    if not osp.exists(dataset_meta_path):
        with open(dataset_meta_path, 'w') as f:
            json.dump(dict(), f)
    return dataset_meta_path


def get_ggl_path():
    global ggl_path
    return ggl_path


global_config_init()


# toolkit
def md5sum(filename):
    """Calculate the File MD5"""
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            md5.update(chunk)
    return md5.hexdigest()


def md5folder(folder):
    """ Calculate Dataset Dir MD5"""
    md5 = hashlib.md5()
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            md5.update(md5sum(filepath).encode('utf-8'))
    return md5.hexdigest()
