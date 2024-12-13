## Overview

In order for users to clearly understand the details of operator testing, we have compiled this document to help users understand the details of the test data set and parameter settings. 

## Dataset

You need to run the `test.py` in the `edge_index/` path to download the dataset. The location of the dataset is `GammaGL/profiler/mpops/edge_index`, `cora.npy` corresponds to the `Cora` dataset, `pubmed.npy` corresponds to the `PubMed` dataset, `ogbn-arxiv.npy` corresponds to the `Ogbn-Arxiv` dataset. 

You can use `np.load(path_to_the_dataset)` to load the corresponding dataset.

The information of the dataset is as follows:

|  Dataset   | Number of nodes | Number of edges |
| :--------: | :-------------: | :-------------: |
|    Cora    |      2708       |      13264      |
|   PubMed   |      19717      |     108368      |
| Ogbn-Arxiv |     169343      |     2315598     |

## Directory of the test code folder

The location of the test code is `GammaGL/profiler/mpops/complete_test`. The following is the directory structure of the `complete_test` folder, `:` followed by the file description. 

```
complete_test
|-- README.md
|-- mp_cpu
|   |-- dgl_mp_cpu.py
|   |-- ggl_mp_cpu.py
|   |-- pd_ext_sum_cpu.py
|   |-- pyg_mp_cpu.py
|   `-- spmm_sum_cpu.py
|-- mp_gpu
|   |-- dgl_mp_gpu.py
|   |-- ggl_mp_gpu.py
|   |-- pd_ext_sum_gpu.py
|   |-- pyg_mp_gpu.py
|   `-- spmm_sum_gpu.py
|-- ops_cpu
|   |-- ggl_segment_cpu.py
|   |-- pd_ext_segment_sum_cpu.py
|   `-- pyg_scatter_ops_cpu.py
`-- ops_gpu
    |-- ggl_segment_gpu.py
    |-- pd_ext_segment_sum_gpu.py
    `-- pyg_scatter_ops_gpu.py
```

## How to run test code

Let's take `ggl_segment_gpu.py` as an example. 

You can switch the program running position through `os.environ["CUDA_VISIBLE_DEVICES"] = "4"` , the `4` represents the fourth GPU in your server, and if you set it to `-1` , it will run on the CPU. 



```python
relative_path = 'profiler/mpops/edge_index/'
file_name = ['cora.npy', 'pubmed.npy', 'ogbn-arxiv.npy']
embedding = [16, 64, 256]
iter = 10
```

You can set the dataset to run by changing the list of `file_name` .

`embedding` is a dimension list representing node feature, and you can set the edbedding dimension by changing the list of `embedding`.

In order to accurately calculate the running time, we increase the running times of each operator. After getting the running time, we divide it by the running times to get the single running time of the operator. `iter` represents the number of times each operator runs.

For `ggl_segment_gpu.py` , you can run it using:

```bash
python ggl_segment_gpu.py
```

