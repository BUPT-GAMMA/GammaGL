Principal Neighbourhood Aggregation for Graph Nets (PNA)
============

- Paper link: [https://grlplus.github.io/papers/20.pdf](https://grlplus.github.io/papers/20.pdf)
- Author's code repo (in PyTorch):
  [https://github.com/lukecavabarrett/pna](https://github.com/lukecavabarrett/pna).

Dataset
-------

The ZINC dataset from the [ZINC database](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559) and the
[Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules](https://arxiv.org/abs/1610.02415) 
paper, containing about 250,000 molecular graphs with up to 38 heavy atoms. Our experiments only load a subset of the 
dataset (12,000 molecular graphs), following the [Benchmarking Graph Neural Networks](https://arxiv.org/abs/2003.00982)
paper.

Results from the Paper
-------

| Task             | Dataset | Model | Metric Name | Metric Value |
|------------------|---------|-------|-------------|--------------|
| Graph Regression | ZINC    | PNA   | MAE         | 0.188±0.004  |


Our Results
-----------

```bash
TL_BACKEND="paddle" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
TL_BACKEND="torch" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
TL_BACKEND="tensorflow" python pna_trainer.py --batch_size 128 --lr 0.001 --n_epoch 400
```

| Dataset | Our(pd)  | Our(torch) | Our(tf)       |
|---------|----------|------------|---------------|
| ZINC    | OOM      | 0.186      | 0.195(±0.006) |


Problems
-----
1. Under the Torch backend, the program runs extremely slow when the convolutional layer gets batch-wise 
graph-level-outputs (Line 160, 163, 166, 169, 174, 175 in **pna_conv.py**). By replacing the corresponding operation with 
function **scatter()** in **torch_scatter.scatter**, the program runs much faster.


2. Under the Paddle backend, the PNA runs erratically using the GPU. The error message is as follows:
```bash
D:\Anaconda3\envs\ggl\python.exe E:/GithubProject/GammaGL/examples/pna/pna_trainer.py
Using Paddle backend.
D:\Anaconda3\envs\ggl\lib\site-packages\flatbuffers\compat.py:19: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
  import imp
W0905 09:55:53.488632  9744 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 11.6, Runtime API Version: 11.6
W0905 09:55:53.488632  9744 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
Error: ../paddle/phi/kernels/funcs/gather.cu.h:67 Assertion `index_value >= 0 && index_value < input_dims[j]` failed. The index is out of bounds, please check whether the dimensions of index and input meet the requirements. It should be less than [56] and greater than or equal to 0, but received [0]
Traceback (most recent call last):
  File "E:\GithubProject\GammaGL\examples\pna\pna_trainer.py", line 137, in <module>
    main(args)
  File "E:\GithubProject\GammaGL\examples\pna\pna_trainer.py", line 65, in main
    d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=tlx.int64)
  File "E:\GithubProject\GammaGL\gammagl\utils\degree.py", line 22, in degree
    return tlx.unsorted_segment_sum(one, index, N)
  File "D:\Anaconda3\envs\ggl\lib\site-packages\tensorlayerx\backend\ops\paddle_backend.py", line 1744, in unsorted_segment_sum
    a = pd.sum(x[segment_ids == i], axis=0)
  File "D:\Anaconda3\envs\ggl\lib\site-packages\paddle\fluid\dygraph\varbase_patch_methods.py", line 736, in __getitem__
    return _getitem_impl_(self, item)
  File "D:\Anaconda3\envs\ggl\lib\site-packages\paddle\fluid\variable_index.py", line 431, in _getitem_impl_
    return get_value_for_bool_tensor(var, slice_item)
  File "D:\Anaconda3\envs\ggl\lib\site-packages\paddle\fluid\variable_index.py", line 309, in get_value_for_bool_tensor
    return cond(
  File "D:\Anaconda3\envs\ggl\lib\site-packages\paddle\fluid\layers\control_flow.py", line 2452, in cond
    pred = pred.numpy()[0]
OSError: (External) CUDA error(719), unspecified launch failure. 
  [Hint: 'cudaErrorLaunchFailure'. An exception occurred on the device while executing a kernel. Common causes include dereferencing an invalid device pointerand accessing out of bounds shared memory. Less common cases can be system specific - more information about these cases canbe found in the system specific user guide. This leaves the process in an inconsistent state and any further CUDA work willreturn the same error. To continue using CUDA, the process must be terminated and relaunched.] (at ..\paddle\phi\backends\gpu\cuda\cuda_info.cc:258)


Process finished with exit code 1
```
One possible reason is that I changed the type of **slice_dict** and **inc_dict** to **dict** (Line 60 in collate.py). 
But if I don't change it, it will cause errors in data preprocessing.