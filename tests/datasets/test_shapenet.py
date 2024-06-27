import json
import os
import os.path as osp
import shutil
import unittest

import tensorlayerx as tlx
from gammagl.data import (
    Graph,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from gammagl.io import read_txt_array
from gammagl.datasets.shapenet import ShapeNet
def test_shapenet():
   class TestShapeNet(unittest.TestCase):

        def setUp(self):
            # 设置数据集根目录和类别，用于测试
            self.dataset = ShapeNet(root='./temp', categories='Airplane')

        def test_download(self):
            # 测试下载方法是否正确
            self.dataset.download()
            raw_files = [os.path.join(self.dataset.raw_dir, fname) for fname in self.dataset.raw_file_names]
            for file in raw_files:
                self.assertTrue(os.path.exists(file))

        def test_process(self):
            # 模拟下载方法
            self.dataset.download = lambda: None

            # 创建一些模拟数据用于测试
            mock_data_dir = os.path.join(self.dataset.raw_dir, '02691156')
            os.makedirs(mock_data_dir, exist_ok=True)
            mock_file = os.path.join(mock_data_dir, 'mock_data.txt')
            with open(mock_file, 'w') as f:
                f.write("0.1 0.2 0.3 0.4 0.5 0.6 0\n0.6 0.5 0.4 0.3 0.2 0.1 1")

            # 创建模拟的分割文件列表
            split_dir = os.path.join(self.dataset.raw_dir, 'train_test_split')
            os.makedirs(split_dir, exist_ok=True)
            with open(os.path.join(split_dir, 'shuffled_train_file_list.json'), 'w') as f:
                json.dump(["02691156/mock_data"], f)

            # 处理模拟数据
            self.dataset.process()

            # 加载处理后的数据
            data, slices = self.dataset.load_data(self.dataset.processed_paths[0])

            # 检查数据是否正确处理
            self.assertEqual(len(data), 1)
            self.assertAlmostEqual(data[0].pos[0].item(), 0.1)
            self.assertAlmostEqual(data[0].y[0].item(), 0)

        def test_repr(self):
            # 测试 __repr__ 方法
            self.assertEqual(repr(self.dataset), 'ShapeNet(1, categories=[\'Airplane\'])')  # 根据数据集调整期望值