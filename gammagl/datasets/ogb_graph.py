import pandas as pd
import shutil, os
import os.path as osp
import numpy as np
from gammagl.data import InMemoryDataset
from gammgl.utils.ogb_url import decide_download, download_url, extract_zip
from gammagl.io.read_ogb import read_graph


class OgbGraphDataset(InMemoryDataset):
    def __init__(self, name, root = 'dataset', transform=None, pre_transform = None, meta_dict = None):
        '''
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - transform, pre_transform (optional): transform/pre-transform graph objects

            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.
        ''' 

        self.name = name ## original name, e.g., ogbg-molhiv
        
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            master = pd.read_csv(os.path.join(os.path.dirname(__file__), 'OgbGraphData.csv'), index_col = 0)
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
        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)

        self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        self.num_tasks = int(self.meta_info['num tasks'])
        self.eval_metric = self.meta_info['eval metric']
        self.task_type = self.meta_info['task type']
        self.__num_classes__ = int(self.meta_info['num classes'])
        self.binary = self.meta_info['binary'] == 'True'

        super(OgbGraphDataset, self).__init__(self.root, transform, pre_transform)

        self.data, self.slices = self.load_data(self.processed_paths[0])

    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return self.load_data(os.path.join(path, 'split_dict.pt'))

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
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        url = self.meta_info['url']
        if decide_download(url):
            path = download_url(url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
            shutil.rmtree(self.root)
            shutil.move(osp.join(self.original_root, self.download_name), self.root)

        else:
            print('Stop downloading.')
            shutil.rmtree(self.root)
            exit(-1)

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

        data_list = read_graph(self.raw_dir, add_inverse_edge = add_inverse_edge, additional_node_files = additional_node_files, additional_edge_files = additional_edge_files, binary=self.binary)

        if self.task_type == 'subtoken prediction':
            graph_label_notparsed = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values
            graph_label = [str(graph_label_notparsed[i][0]).split(' ') for i in range(len(graph_label_notparsed))]

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        else:
            if self.binary:
                graph_label = np.load(osp.join(self.raw_dir, 'graph-label.npz'))['graph_label']
            else:
                graph_label = pd.read_csv(osp.join(self.raw_dir, 'graph-label.csv.gz'), compression='gzip', header = None).values

            has_nan = np.isnan(graph_label).any()

            for i, g in enumerate(data_list):
                g.y = graph_label[i]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        self.save_data((data, slices), self.processed_paths[0])


