## Overview

In order for users to clearly understand the details of operator testing, we have compiled this document to help users understand the details of the test data set and parameter settings. 

## Dataset

The location of the dataset is `GammaGL/profiler/mpops/edge_index`, `cora.npy` corresponds to the `Cora` dataset, `pubmed.npy` corresponds to the `PubMed` dataset, `ogbn-arxiv.npy` corresponds to the `Ogbn-Arxiv` dataset. 

You can use `np.load(path_to_the_dataset)` to load the corresponding dataset.

The information of the dataset is as follows:

|  Dataset   | Number of nodes | Number of edges |
| :--------: | :-------------: | :-------------: |
|    Cora    |      2708       |      13264      |
|   PubMed   |      19717      |     108368      |
| Ogbn-Arxiv |     169344      |     2315598     |

## Directory of the test code folder

The location of the test code is `GammaGL/profiler/mpops/complete_test`. The following is the directory structure of the `complete_test` folder, `:` followed by the file description. 

```
complete_test
├── mp_cpu: code folder to test the efficiency of message passing process under CPU
│   ├── dgl_mp_cpu.py
│   ├── ms_mp_cpu.py
│   ├── pd_ext_sum_cpu.py
│   ├── pd_mp_cpu.py
│   ├── pyg_mp_cpu.py
│   ├── spmm_sum_cpu.py
│   ├── tf_mp_cpu.py
│   ├── th_ext_max_cpu.py
│   └── th_mp_cpu.py
├── mp_gpu: code folder to test the efficiency of message passing process under GPU
│   ├── dgl_mp_gpu.py
│   ├── ms_mp_gpu.py
│   ├── pd_ext_sum_gpu.py
│   ├── pd_mp_gpu.py
│   ├── pyg_mp_gpu.py
│   ├── spmm_sum_gpu.py
│   ├── tf_mp_gpu.py
│   ├── th_ext_max_gpu.py
│   └── th_mp_gpu.py
├── ops_cpu: code folder for testing the efficiency of operators under CPU
│   ├── ms_segment_ops_cpu.py
│   ├── pd_ext_segment_sum_cpu.py
│   ├── pd_segment_ops_cpu.py
│   ├── pyg_scatter_ops_cpu.py
│   ├── tf_segment_ops_cpu.py
│   ├── th_ext_segment_max_cpu.py
│   └── th_segment_ops_cpu.py
└── ops_gpu: code folder for testing the efficiency of operators under GPU
	├── ms_segment_ops_gpu.py
    ├── pd_ext_segment_sum_gpu.py
    ├── pd_segment_ops_gpu.py
    ├── pyg_scatter_ops_gpu.py
    ├── tf_segment_ops_gpu.py
    ├── th_ext_segment_max_gpu.py
    └── th_segment_ops_gpu.py
```

## How to run test code

Let's take `th_segment_ops_gpu.py` as an example. 

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

For `th_segment_ops_gpu.py` , you can run it using:

```bash
python th_segment_ops_gpu.py
```

