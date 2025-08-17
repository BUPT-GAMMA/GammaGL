import argparse
import random
import sys
import os
import numpy as np

os.environ['TL_BACKEND'] = 'torch'  

import tensorlayerx as tlx
import logging
from datetime import datetime
from utils import Config, setup_logger

from runner_base import RunnerBase
from arxiv_text_pair_datasets import ArxivTextPairDataset
from gammagl.models import TranslatorCHATGLMArxiv, TranslatorQformerArxiv


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--cfg-path", default="./pretrain_arxiv_stage1.yaml", help="path to configuration file.")
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


def main(job_id):
    cfg = Config(parse_args())

    setup_seeds(cfg)
    setup_logger()
    cfg.pretty_print()


    datasets = build_datasets(cfg)
    if cfg.model_cfg.arch == "translator_arxiv_chatglm":
        model = TranslatorCHATGLMArxiv.from_config(cfg.model_cfg)
    elif cfg.model_cfg.arch == "translator_arxiv":
        model = TranslatorQformerArxiv.from_config(cfg.model_cfg)


    runner = RunnerBase(
        cfg=cfg, job_id=job_id, model=model, datasets=datasets)
    runner.train()


if __name__ == "__main__":
    job_id = datetime.now().strftime("%Y%m%d%H%M")[:-1]
    main(job_id)
