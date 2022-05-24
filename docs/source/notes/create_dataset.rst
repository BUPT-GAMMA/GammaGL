Creating Your Own Datasets
==========================
We follow the `torch_geometric.data.Dataset <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html>`_ module and create the :class:`gammagl.data.Dataset` with a few modifications.

Although GammaGL already contains a lot of useful datasets, you may wish to create your own dataset with self-recorded or non-publicly available data.

Implementing datasets by yourself is straightforward and you may want to take a look at the source code to find out how the various datasets are implemented.
However, we give a brief introduction on what is needed to setup your own dataset.

We provide two abstract classes for datasets: :class:`gammagl.data.Dataset` and :class:`gammagl.data.InMemoryDataset`.
:class:`gammagl.data.InMemoryDataset` inherits from :class:`gammagl.data.Dataset` and should be used if the whole dataset fits into CPU memory.

Following the :obj:`torchvision` convention, each dataset gets passed a root folder which indicates where the dataset should be stored.
We split up the root folder into two folders: the :obj:`raw_dir`, where the dataset gets downloaded to, and the :obj:`processed_dir`, where the processed dataset is being saved.

In addition, each dataset can be passed a :obj:`transform`, a :obj:`pre_transform` and a :obj:`pre_filter` function, which are :obj:`None` by default.
The :obj:`transform` function dynamically transforms the data object before accessing (so it is best used for data augmentation).
The :obj:`pre_transform` function applies the transformation before saving the data objects to disk (so it is best used for heavy precomputation which needs to be only done once).
The :obj:`pre_filter` function can manually filter out data objects before saving.
Use cases may involve the restriction of data objects being of a specific class.

Creating "In Memory Datasets"
-----------------------------

In order to create a :class:`gammagl.data.InMemoryDataset`, you need to implement four fundamental methods:

* :func:`gammagl.data.InMemoryDataset.raw_file_names`: A list of files in the :obj:`raw_dir` which needs to be found in order to skip the download.

* :func:`gammagl.data.InMemoryDataset.processed_file_names`: A list of files in the :obj:`processed_dir` which needs to be found in order to skip the processing.
  GammaGL recommends setting it with :obj:`tlx.BACKEND + '_data.pt'` due to involving multi-backends.

* :func:`gammagl.data.InMemoryDataset.download`: Downloads raw data into :obj:`raw_dir`.

* :func:`gammagl.data.InMemoryDataset.process`: Processes raw data and saves it into the :obj:`processed_dir`.

You can find helpful methods to download and extract data in :mod:`gammagl.data`.

The real magic happens in the body of :meth:`~gammagl.data.InMemoryDataset.process`.
Here, we need to read and create a list of :class:`~gammagl.data.Graph` objects and save it into the :obj:`processed_dir`.
Because saving a huge python list is rather slow, we collate the list into one huge :class:`~gammagl.data.Graph` object via :meth:`gammagl.data.InMemoryDataset.collate` before saving.
The collated data object has concatenated all examples into one big data object and, in addition, returns a :obj:`slices` dictionary to reconstruct single examples from this object.
Finally, we need to load these two objects in the constructor into the properties :obj:`self.data` and :obj:`self.slices`.

Let's see this process in a simplified example:

.. code-block:: python

    import tensorlayerx as tlx
    from gammagl.data import InMemoryDataset, download_url


    class MyOwnDataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = self.load_data(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return ['some_file_1', 'some_file_2', ...]

        @property
        def processed_file_names(self):
            return tlx.BACKEND + '_data.pt

        def download(self):
            # Download to `self.raw_dir`.
            download_url(url, self.raw_dir)
            ...

        def process(self):
            # Read data into huge `Data` list.
            data_list = [...]

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data, slices = self.collate(data_list)
            self.save_data((data, slices), self.processed_paths[0])

Creating "Larger" Datasets
--------------------------

For creating datasets which do not fit into memory, the :class:`gammagl.data.Dataset` can be used, which closely follows the concepts of the :obj:`torchvision` datasets.
It expects the following methods to be implemented in addition:

* :func:`gammagl.data.Dataset.len`: Returns the number of examples in your dataset.

* :func:`gammagl.data.Dataset.get`: Implements the logic to load a single graph.

Internally, :meth:`gammagl.data.Dataset.__getitem__` gets data objects from :meth:`gammagl.data.Dataset.get` and optionally transforms them according to :obj:`transform`.

Let's see this process in a simplified example:

.. code-block:: python

    import os.path as osp

    import tensorlayerx as tlx
    from gammagl.data import Dataset, download_url


    class MyOwnDataset(Dataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)

        @property
        def raw_file_names(self):
            return ['some_file_1', 'some_file_2', ...]

        @property
        def processed_file_names(self):
            return ['data_1.pt', 'data_2.pt', ...]

        def download(self):
            # Download to `self.raw_dir`.
            path = download_url(url, self.raw_dir)
            ...

        def process(self):
            idx = 0
            for raw_path in self.raw_paths:
                # Read data from `raw_path`.
                data = Graph(...)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                self.save_data((data, _), osp.join(self.processed_dir, tlx.BACKEND + f'data_{idx}.pt'))
                idx += 1

        def len(self):
            return len(self.processed_file_names)

        def get(self, idx):
            data, _ = self.load_data(osp.join(self.processed_dir, tlx.BACKEND + f'data_{idx}.pt'))
            return data

Here, each graph data object gets saved individually in :meth:`~gammagl.data.Dataset.process`, and is manually loaded in :meth:`~gammagl.data.Dataset.get`.

Frequently Asked Questions
--------------------------

#. **How can I skip the execution of** :meth:`download` **and/or** :meth:`process` **?**

    You can skip downloading and/or processing by just not overriding the :meth:`download()` and :meth:`process()` methods:

    .. code-block:: python

        class MyOwnDataset(Dataset):
            def __init__(self, transform=None, pre_transform=None):
                super().__init__(None, transform, pre_transform)

#. **Do I really need to use these dataset interfaces?**

    No! Just as in regular PyTorch, you do not have to use datasets, *e.g.*, when you want to create synthetic data on the fly without saving them explicitly to disk.
    In this case, simply pass a regular python list holding :class:`gammagl.data.Data` objects and pass them to :class:`gammagl.loader.DataLoader`:

    .. code-block:: python

        from gammagl.data import Data
        from gammagl.loader import DataLoader

        data_list = [Data(...), ..., Data(...)]
        loader = DataLoader(data_list, batch_size=32)

#. **How I build dataset and integrate it into GammaGL?**

    Besides the above tutorials, note that GammaGL is a multi-backend library. The best way to be compatible with different backend,
    we should process dataset using Numpy or something, which is framework-agnostic. At last, the :obj:`Graph` constructor will get data with :obj:`numpy.array` and modify them into Tensor.

Exercises
---------

Consider the following :class:`~gammagl.data.InMemoryDataset` constructed from a list of :obj:`~gammagl.data.Data` objects:

.. code-block:: python

    class MyDataset(InMemoryDataset):
        def __init__(self, root, data_list, transform=None):
            self.data_list = data_list
            super().__init__(root, transform)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def processed_file_names(self):
            return 'data.pt'

        def process(self):
            torch.save(self.collate(self.data_list), self.processed_paths[0])

