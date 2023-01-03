from time import time
import os

os.environ["TL_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 6'

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gammagl.datasets import Reddit

import numpy as np
import scipy.sparse as sp
from torch.utils.dlpack import from_dlpack

from gammagl.gglspeedup.multifeat import Multi_CGPUFeature
from gammagl.gglspeedup.multigpusample import MultiGPUSampler


def run(rank, world_size, sampler, feat, train_mask):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    train_idx = train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    train_loader = torch.utils.data.DataLoader(train_idx, batch_size=1024, shuffle=True, drop_last=True)
    for i in range(20):
        st = time()
        sampler.set_size([25, 10])
        for seed in train_loader:
            all_node, _ = sampler.sample(seed, rank)
            _ = from_dlpack(feat[all_node].toDlpack())
        ed = time()
        print(ed - st)


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    n_gpus = 2
    world_size = n_gpus

    dataset = Reddit('../../reddit')
    data = dataset[0]

    csr = sp.csr_matrix((np.ones((data.edge_index.shape[1],)), data.edge_index.numpy()))
    sampler = MultiGPUSampler(1, csr)
    multi_feat = Multi_CGPUFeature(0, [0, 1], ".1G", csr)
    multi_feat.from_cpu_tensor(data.x.numpy())

    mp.spawn(run, args=(world_size, sampler, multi_feat, data.train_mask), nprocs=world_size, join=True)
