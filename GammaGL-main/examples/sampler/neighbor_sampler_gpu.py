from gammagl.utils import mask_to_index
from gammagl.datasets import Reddit
import argparse
from pyinstrument import Profiler
import scipy.sparse as sp
import numpy as np
from gammagl.gglspeedup.gpusample import GPUSampler
from gammagl.gglspeedup.gpufeature import CGPUFeature
import tensorflow as tf

dataset = Reddit('../../data/reddit')
graph = dataset[0]

train_idx = mask_to_index(graph.train_mask)


def main(args):

    if args.backend == 'tensorflow':
        print("Using tensorflow backend gpu sampler...")
        from tensorflow.python import from_dlpack
    if args.backend == 'torch':
        print("Using torch backend gpu sampler...")
        from torch.utils.dlpack import from_dlpack
    if args.backend == 'paddle':
        print("Using paddle backend gpu sampler...")
        from paddle.utils.dlpack import from_dlpack

    csr = sp.csr_matrix((np.ones((graph.edge_index.shape[1],)), graph.edge_index.numpy()))

    cgpu_feat = CGPUFeature(csr, args.device, "0.1G")
    cgpu_feat.from_cpu_tensor(graph.x.numpy())
    sampler = GPUSampler(args.device, csr)

    sample_lists = args.sample_lists.split(',')
    for i, num in enumerate(sample_lists):
        sample_lists[i] = int(num)

    sampler.set_size(sample_lists)
    train_mask = graph.train_mask
    train_idx = mask_to_index(train_mask)
    # use tf function to speed up
    dataset = tf.data.Dataset.from_tensor_slices(train_idx)

    train_loader = dataset.shuffle(train_idx.shape[0]).batch(args.batch_size)

    profiler = Profiler()
    profiler.start()
    for seed in train_loader:
        '''use cp2tf'''
        all_node, adjs = sampler.sample(seed)
        cur_feat = cgpu_feat[all_node]

        all_node = from_dlpack(all_node.toDlpack())
        cur_feat = from_dlpack(cur_feat.toDlpack())

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_lists", type=str, default="25,10", help="sample number in each layer")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of target nodes")
    parser.add_argument("--device", type=int, default=3, help="choose gpu device")
    parser.add_argument("--backend", type=str, default="tensorflow", help="choose backend(tensorflow,torch,paddle)")
    args = parser.parse_args()

    main(args)
