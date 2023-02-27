# GPU_Muliti

The idea from [torch-quiver](https://github.com/quiver-team/torch-quiver).

# Installation

you need install CuPy 、Numba、CURandRTC

```
pip install CuPy
conda install Numba
pip install CURandRTC

```

# Designed For GammaGL

This third-party library designed for [GammaGL](https://github.com/BUPT-GAMMAGammaGL) , like [torch-quiver](https://github.com/quiver-team/torch-quiver), it's used for Sampling-Based Graph Learning. It can accelerate sampling neighbor nodes and get node's feature speed. And it like tensorlayerx, suppport multiple deep learning backends.

## additions

This third-party library also support graph learning library developed based on other deep learning backends. Because many Deep Learning backends' Tensor support DLPack, also CuPy is support too. So, take advantage of DLPack's  high-speed conversion backend.

# Use demo

### GammaGL:

###### use Reddit and tensorlayerx use PyTorch backend as example

```python
path = '../reddit'
dataset = Reddit(path)
data = dataset[0]
rank = 0
csr = sp.csr_matrix((np.ones((data.edge_index.shape[1],)), data.edge_index.numpy()))
...
cgpu_feat = CGPUFeature(csr, rank, "0.1G")
cgpu_feat.from_cpu_tensor(data.x.numpy())
sampler = GPUSampler(rank, csr)
sampler.set_size([25, 10])
...
for seed in train_loader:
    all_node, adjs = sampler.sample(seed)
    cur_feat = cgpu_feat[all_node]
    '''use cp2th'''
    adjs = [adj.cp2torch() for adj in adjs]
    all_node = from_dlpack(all_node.toDlpack())
    cur_feat = from_dlpack(cur_feat.toDlpack())
    data = {"x": cur_feat,
            "y": y,
            "dst_node": all_node[:seed.shape[0]],
            "subgs": adjs}
    # label is not used
    train_loss = train_one_step(data, tlx.convert_to_tensor([0]))
```



### other graph learning library:

```python
graph = ...
import scipy.sparse as sp
csr = sp.csr_matrix((np.ones((graph.edge_index.shape[1],)), data.edge_index.numpy()))

from gglspeedup.gpusample import GPUSampler
from gglspeedup.gpufeature import CGPUFeature

cgpu_feat = CGPUFeature(csr, 0, "0.1G")
cgpu_feat.from_cpu_tensor(data.feature.numpy())
sampler = GPUSampler(0, csr)
for seeds in train_loader:
    all_node, adjs = sampler.sample(seed)
    cur_feat = cgpu_feat[all_node]
	
    # this part you can shuffle Deep Learning backends
    # we use tensorflow backend as demo
    
    from tensorflow.python import from_dlpack
    adjs = [adj.cp2tf() for adj in adjs]
    all_node = from_dlpack(all_node.toDlpack())
    cur_feat = from_dlpack(cur_feat.toDlpack())

```

### Multi-GPU:
You need to read this ([predo.md](demoulti/predo.md)) first.

Because TensorLayerx don't support multiprocessing train. Thus, use the original API of deep learing backend to do distributed graph learning. This part use paddle to show the demo.

```
def train(ml_sp, feat):
    dist.init_parallel_env()

    train_loader = tlx.dataflow.DataLoader(tlx.arange(0, 200000, dtype=paddle.int64), batch_size=1024, shuffle=True)
    ml_sp.set_size([25, 10])
    model = ...
    for i in range(1, 20):
        for seed in train_loader:
            seed = tlx.reshape(seed, (-1,))
            dst_node, cadjs = ml_sp.sample(seed, dist.get_rank())
            cur_feat = feat[dst_node]
            data = {"x": cur_feat,
                    "y": y,
                    "dst_node": all_node[:seed.shape[0]],
                    "subgs": adjs}
            train_loss = train_one_step(data, tlx.convert_to_tensor([0]))
            ....

if __name__ == '__main__':
    ml_sp = MultiGPUSampler(1, csr)
    multi_feat = Multi_CGPUFeature(1, [0, 1], ".1G", csr)
    multi_feat.from_cpu_tensor(data.x.numpy())
    dist.spawn(train, args=(ml_sp, ), nprocs=2, gpus='0,1')
```

# TODO in Future:

- paddle backend meet a bug in using from_dlpack, it will costs oom, you can try in demo(), so paddle backend support multi-gpus sample and single-gpu sample now. We will make it work when paddle fix the bug.

- paddle 's dataloader is very slow, it will make speed test inaccurately. And I have no idea to accelerate it.

- When Tensorlayerx in TensorFlow backend，tlx' s dataloader is slow too, so this part, you can choose use this code to in place tlx's dataloader

  ```python
  dataset = tf.data.Dataset.from_tensor_slices(train_idx)
  train_loader = dataset.shuffle(train_idx.shape[0]).batch(batch_size)
  ```

- Make distributed graph learning in using tensorflow backend possible.

 
