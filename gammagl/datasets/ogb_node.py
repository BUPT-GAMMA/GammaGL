import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
from gammagl.data import InMemoryDataset
from gammagl.data.download import download_url
from gammagl.data.extract import extract_zip
from gammagl.io.read_ogb import read_node_label_hetero, read_graph, read_heterograph,read_nodesplitidx_split_hetero


class OgbNodeDataset(InMemoryDataset):
    def __init__(self, name, root='dataset', transform=None, pre_transform=None, meta_dict=None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        '''

        self.name = name  ## original name, e.g., ogbn-proteins

        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-'))

            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_gammagl')):
                self.dir_name = self.dir_name + '_gammagl'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)

            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'OgbNodeData.csv'), index_col=0)
            if not self.name in master:
                error_mssg = 'Invalid dataset name {}.\n'.format(self.name)
                error_mssg += 'Available datasets are as follows:\n'
                error_mssg += '\n'.join(master.keys())
                raise ValueError(error_mssg)
            self.meta_info = master[self.name]

        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict

        # check version
        # First check whether the dataset has been already downloaded or not.
        # If so, check whether the dataset version is the newest or not.
        # If the dataset is not the newest version, notify this to the user.
        if osp.isdir(self.root) and (
        not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name']  ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.task_type = self.meta_info['task type']
        self.eval_metric = self.meta_info['eval metric']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.is_hetero = self.meta_info['is hetero'] == 'True'
        self.binary = self.meta_info['binary'] == 'True'

        super(OgbNodeDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = self.load_data(self.processed_paths[0])
    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']

        path = osp.join(self.root, 'split', split_type)

        if self.is_hetero:
            train_idx_dict, valid_idx_dict, test_idx_dict = read_nodesplitidx_split_hetero(path)
            for nodetype in train_idx_dict.keys():
                train_idx_dict[nodetype] = train_idx_dict[nodetype]
                valid_idx_dict[nodetype] = valid_idx_dict[nodetype]
                test_idx_dict[nodetype] = test_idx_dict[nodetype]

                return {'train': train_idx_dict, 'valid': valid_idx_dict, 'test': test_idx_dict}

        else:
            train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
            valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
            test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

            return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    @property
    def num_classes(self):
        return self.__num_classes__

    @property
    def raw_file_names(self):
        if self.binary:
            if self.is_hetero:
                return ['edge_index_dict.npz']
            else:
                return ['data.npz']
        else:
            if self.is_hetero:
                return ['num-node-dict.csv.gz', 'triplet-type-list.csv.gz']
            else:
                file_names = ['edge']
                if self.meta_info['has_node_attr'] == 'True':
                    file_names.append('node-feat')
                if self.meta_info['has_edge_attr'] == 'True':
                    file_names.append('edge-feat')
                return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return osp.join('geometric_data_processed.pt')

    def download(self):
        url = self.meta_info['url']
        path = download_url(url, self.original_root)
        extract_zip(path, self.original_root)
        os.unlink(path)
        shutil.rmtree(self.root)
        shutil.move(osp.join(self.original_root, self.download_name), self.root)


    def process(self):
        add_inverse_edge = self.meta_info['add_inverse_edge'] == 'True'

        if self.meta_info['additional node files'] == 'None':
            additional_node_files = []
        else:
            additional_node_files = self.meta_info['additional node files'].split(',')

        if self.meta_info['additional edge files'] == 'None':
            additional_edge_files = []
        else:
            additional_edge_files = self.meta_info['additional edge files'].split(',')

        if self.is_hetero:
            data = read_heterograph(self.raw_dir, add_inverse_edge=add_inverse_edge,
                                    additional_node_files=additional_node_files,
                                    additional_edge_files=additional_edge_files, binary=self.binary)[0]

            if self.binary:
                tmp = np.load(osp.join(self.raw_dir, 'node-label.npz'))
                node_label_dict = {}
                for key in list(tmp.keys()):
                    node_label_dict[key] = tmp[key]
                del tmp
            else:
                node_label_dict = read_node_label_hetero(self.raw_dir)

            data.y_dict = {}
            if 'classification' in self.task_type:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = node_label
            else:
                for nodetype, node_label in node_label_dict.items():
                    data.y_dict[nodetype] = node_label
        else:
            data = \
            read_graph(self.raw_dir, add_inverse_edge=add_inverse_edge, additional_node_files=additional_node_files,
                           additional_edge_files=additional_edge_files, binary=self.binary)[0]
            ### adding prediction target
            if self.binary:
                node_label = np.load(osp.join(self.raw_dir, 'node-label.npz'))['node_label']
            else:
                node_label = pd.read_csv(osp.join(self.raw_dir, 'node-label.csv.gz'), compression='gzip',
                                         header=None).values
            data.y = node_label
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.data = data
        print('Saving...')
        self.save_data(self.collate([data]), self.processed_paths[0])

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    data = OgbNodeDataset(name='ogbn-arxiv')
    print(data[0])
