import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from gammagl.gglspeedup.gpusample import GPUSampler
from gammagl.gglspeedup.gpufeature import CGPUFeature
from tensorflow.python import from_dlpack

import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
for gpu in devices:
    tf.config.experimental.set_memory_growth(gpu, True)


from gammagl.utils import mask_to_index
from gammagl.datasets import Reddit

import numpy as np
import scipy.sparse as sp
from time import time





if __name__ == "__main__":
    path = '../reddit'
    dataset = Reddit(path)
    data = dataset[0]
    csr = sp.csr_matrix((np.ones((data.edge_index.shape[1],)), data.edge_index.numpy()))

    cgpu_feat = CGPUFeature(csr, 0, "0.1G")
    cgpu_feat.from_cpu_tensor(data.x.numpy())
    sampler = GPUSampler(0, csr)
    sampler.set_size([25, 10])
    train_mask = data.train_mask
    train_idx = mask_to_index(train_mask)

    dataset = tf.data.Dataset.from_tensor_slices(train_idx)
    train_loader = dataset.shuffle(train_idx.shape[0]).batch(1024)

    for i in range(20):
        st = time()
        print("sampling ...")
        total_loss = 0
        total_correct = 0
        for seed in train_loader:
            '''use cp2tf'''
            all_node, adjs = sampler.sample(seed)
            cur_feat = cgpu_feat[all_node]
            all_node = from_dlpack(all_node.toDlpack())
            cur_feat = from_dlpack(cur_feat.toDlpack())

        ed = time()
        print(ed - st)
        print("=" * 300)