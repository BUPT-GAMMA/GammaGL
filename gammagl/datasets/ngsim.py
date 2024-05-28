import argparse
import os.path
import sys
import zipfile
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import requests
import tensorlayerx as tlx
from gammagl.data import Graph, HeteroGraph, download_url, extract_zip
from gammagl.data import Dataset
import random
import shutil

os.environ['TL_BACKEND'] = 'torch'

device = tlx.set_device(device='GPU', id=0)
# device = tlx.set_device(device='cpu')


class process_data():
    def __init__(self, save_path='.', hist_len=10, fut_len=50, url=''):
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.save_path = save_path
        self.selected_data = pd.read_csv(url)
        self.node_type_to_indicator_vec = {1: tlx.convert_to_tensor([[0, 0, 1]]),
                                           2: tlx.convert_to_tensor([[0, 1, 0]]),
                                           3: tlx.convert_to_tensor([[0, 1, 0]])}

    def get_veh_id_to_traj(self):
        veh_id_to_adj = {}
        veh_IDs = set(self.selected_data['Vehicle_ID'].values)
        for i, veh_id in enumerate(veh_IDs):
            veh_traj = self.selected_data[self.selected_data['Vehicle_ID'] == veh_id][
                ['Vehicle_ID', 'Frame_ID', 'Global_X', 'Global_Y', 'v_Class', 'v_Vel']]
            veh_id_to_adj[veh_id] = veh_traj
        return veh_id_to_adj

    def get_hist(self, vehicle_id, cur_time):
        cur_frame_id = self.selected_data[self.selected_data['Global_Time'] == cur_time]['Frame_ID'].values[0]
        # print(cur_frame_id)
        start_frm_id = cur_frame_id - self.hist_len + 1
        traj = self.get_veh_id_to_traj()[vehicle_id]

        raw_hist = traj[(traj['Frame_ID'] >= start_frm_id) & (traj['Frame_ID'] <= cur_frame_id)][
            ['Global_X', 'Global_Y', 'v_Vel']].values
        if raw_hist.shape[0] > self.hist_len:
            raw_hist = raw_hist[:self.hist_len]

        new_hist = raw_hist.copy()
        new_hist[:, :2] = 0
        veh_psi = np.zeros((new_hist.shape[0], 1))
        new_hist = np.concatenate((new_hist, veh_psi), axis=1)

        # print(cur_frame_id, new_hist.shape[0], raw_hist.shape[0])
        if new_hist.shape[0] < self.hist_len:
            new_hist = np.pad(new_hist, pad_width=((self.hist_len - new_hist.shape[0], 0), (0, 0)), mode='empty')
            # print("new_new", new_hist.shape[0])
        if raw_hist.shape[0] < self.hist_len:
            raw_hist = np.pad(raw_hist, pad_width=((self.hist_len - raw_hist.shape[0], 0), (0, 0)), mode='empty')
            # print("new_raw", raw_hist.shape[0])
        return new_hist, raw_hist

    def get_fut(self, veh_id, cur_time):
        be_target = True
        cur_frm_id = self.selected_data[self.selected_data['Global_Time'] == cur_time]['Frame_ID'].values[0]
        fur_frm = cur_frm_id + self.fut_len
        traj = self.get_veh_id_to_traj()[veh_id]
        raw_fut = traj[(traj['Frame_ID'] > cur_frm_id) & (traj['Frame_ID'] <= fur_frm)][['Global_X', 'Global_Y']].values
        if raw_fut.shape[0] > self.fut_len:
            raw_fut = raw_fut[:self.fut_len]

        new_fut = raw_fut.copy()
        new_fut[:, :2] = 0
        # veh_psi = np.zeros((new_fut.shape[0], 1))
        # new_fut = np.concatenate((new_fut, veh_psi), axis=1)
        # print(cur_frm_id, new_fut.shape[0], raw_fut.shape[0])
        if new_fut.shape[0] < self.fut_len:
            be_target = False
            new_fut = np.pad(new_fut, pad_width=((self.fut_len - new_fut.shape[0], 0), (0, 0)), mode='constant',
                             constant_values=0)
            # print( "new-new:", new_fut.shape[0])
        if raw_fut.shape[0] < self.fut_len:
            raw_fut = np.pad(raw_fut, pad_width=((self.fut_len - raw_fut.shape[0], 0), (0, 0)), mode='constant',
                             constant_values=0)
            # print("raw_new:", raw_fut.shape[0])
        return new_fut, raw_fut, be_target

    def get_nbrs(self, v_id, cur_time, radii=30):
        Time = self.selected_data[self.selected_data['Global_Time'] == cur_time]

        # current location of target vehicle
        ref_pos = Time[Time['Vehicle_ID'] == v_id][['Global_X', 'Global_Y']].values[0]
        # 计算每辆车和目标车间的欧氏距离
        x_sqr = np.square(Time[['Global_X']].values - ref_pos[0])
        y_sqr = np.square(Time[['Global_Y']].values - ref_pos[1])

        Time.insert(len(Time.columns), 'dist_veh_nbrs', np.sqrt(x_sqr + y_sqr), False)
        return Time[(Time['dist_veh_nbrs'] > 0.01) & (Time['dist_veh_nbrs'] <= radii) & (Time['Vehicle_ID'] != v_id)]['Vehicle_ID'].values

    def process(self):
        time_numbers = list(set(self.selected_data['Global_Time'].values))
        print(self.selected_data['Global_Time'].values)

        num = len(time_numbers)
        for i in time_numbers:
            Time = self.selected_data[self.selected_data['Global_Time'] == i]

            num -= 1
            # print(num)

            raw_v_ids = Time['Vehicle_ID'].values
            # print("raw_v_ids:", raw_v_ids)

            if len(raw_v_ids) == 0:
                continue
            else:
                # 将车辆ID映射到节点索引的字典
                v_id_to_node_index = {}
                for j, v_id in enumerate(raw_v_ids):
                    v_id_to_node_index[v_id] = j
                print(v_id_to_node_index)

                # Node feature
                Nodes_f = tlx.zeros((len(raw_v_ids), self.hist_len, 4)).to(device)
                Fut_GT = tlx.zeros((len(raw_v_ids), self.fut_len, 2)).to(device)

                # Edge
                Edges = tlx.zeros((2, 0)).long().to(device)
                Edges_attr = tlx.zeros((5, 0)).to(device)  # [d_x, d_y, l_x, l_y, d_psi]
                Edges_type = tlx.zeros((6, 0)).long().to(device)

                # Masks
                Tar_Mask = []
                Veh_Tar_Mask = []  # mask for target vehicles
                Veh_Mask = []
                Ped_Mask = []

                # 初始化原始历史轨迹和未来轨迹
                Raw_hist = tlx.zeros((len(raw_v_ids), self.hist_len, 3)).to(device)
                Raw_fut = tlx.zeros((len(raw_v_ids), self.fut_len, 2)).to(device)

                # 对于每个车辆
                for j, v_id in enumerate(raw_v_ids):
                    # node feature and hist_traj
                    v_hist, raw_h = self.get_hist(v_id, i)
                    Nodes_f[j] = tlx.convert_to_tensor(v_hist)
                    Raw_hist[j] = tlx.convert_to_tensor(raw_h)

                    veh_cur_state = Time[Time['Vehicle_ID'] == v_id][
                        ['Global_X', 'Global_Y', 'Local_X', 'Local_Y', 'v_Vel']].values[0]
                    veh_cur_state = tlx.convert_to_tensor(veh_cur_state)
                    veh_type = Time[Time['Vehicle_ID'] == v_id]['v_Class'].values[0]
                    if veh_type == 1:
                        Veh_Mask.append(False)
                        Ped_Mask.append(True)
                    elif veh_type == 2 or veh_type == 3:
                        Veh_Mask.append(True)
                        Ped_Mask.append(False)

                    # edge
                    # 当前车辆在图中的索引
                    v_node_index = v_id_to_node_index[v_id]

                    # 对于每辆车的邻居车辆：
                    v_nbrs = self.get_nbrs(v_id=v_id, cur_time=i)
                    for v_nbr_id in v_nbrs:
                        print("v_id -> v_nbrs:", v_id, v_nbrs)
                        nbr_v_node_index = v_id_to_node_index[v_nbr_id]

                        nbr_cur_state = Time[Time['Vehicle_ID'] == v_nbr_id][
                            ['Global_X', 'Global_Y', 'Local_X', 'Local_Y', 'v_Vel']].values[0]
                        nbr_cur_state = tlx.convert_to_tensor(nbr_cur_state)
                        nbr_type = Time[Time['Vehicle_ID'] == v_nbr_id]['v_Class'].values[0]

                        # edge
                        edge = tlx.convert_to_tensor([[nbr_v_node_index], [v_node_index]])
                        edge_attr = nbr_cur_state - veh_cur_state
                        edge_attr = edge_attr.float().unsqueeze(axis=1)
                        edge_type = tlx.concat(
                            (self.node_type_to_indicator_vec[nbr_type], self.node_type_to_indicator_vec[nbr_type]),
                            axis=1)
                        Edges = tlx.concat((Edges, edge), axis=1)
                        Edges_attr = tlx.concat((Edges_attr, edge_attr), axis=1)

                        # print("bnrs:", edge)

                        Edges_type = tlx.concat((Edges_type, edge_type.transpose(0, 1)), axis=1)

                    # future trajectories
                    v_fut, raw_fut, tar = self.get_fut(veh_id=v_id, cur_time=i)
                    Fut_GT[j] = tlx.convert_to_tensor(v_fut)
                    Raw_fut[j] = tlx.convert_to_tensor(raw_fut)
                    Tar_Mask.append(tar)

                    if veh_type == 2 and tar:
                        Veh_Tar_Mask.append(True)
                    else:
                        Veh_Tar_Mask.append(False)

                # print(i, raw_v_ids, Edges, Edges_attr.shape)

                Tar_Mask = tlx.convert_to_tensor(Tar_Mask)
                Veh_Tar_Mask = tlx.convert_to_tensor(Veh_Tar_Mask)
                Veh_Mask = tlx.convert_to_tensor(Veh_Mask)
                Ped_Mask = tlx.convert_to_tensor(Ped_Mask)

                if tlx.all(Edges_attr == 0).item():
                    continue

                gl_data = Graph(x=Nodes_f, y=Fut_GT, edge_index=Edges, edge_attr=Edges_attr, edge_type=Edges_type,
                                tar_mask=Tar_Mask, veh_tar_mask=Veh_Tar_Mask, veh_mask=Veh_Mask, ped_mask=Ped_Mask,
                                raw_hist=Raw_hist, raw_fut=Raw_fut)
                gl_data_name = '{}/train_time_{}'.format(self.save_path, i)
                # print(gl_data_name)
                # print(gl_data)
                tlx.files.save_any_to_npy(gl_data, gl_data_name)
                # num += 1
                print(i)
                print("data.edge: ", gl_data.edge_index)
                print("data.edge_attr: ", gl_data.edge_attr)


class NGSIM_US_101(Dataset):
    def __init__(self, data_path, hist_len=10, fut_len=10, save_to=''):
        super(NGSIM_US_101).__init__()
        self.data_path = data_path
        self.hist_len = hist_len
        self.gut_len = fut_len
        self.save_to = save_to
        self.url = 'https://raw.githubusercontent.com/gjy1221/NGSIM-US-101/main/data/data.zip'
        self.data_names = os.listdir('{}'.format(self.data_path))
        print(self.data_path)

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        # name = self.data_names[index]
        # for i in range(len(self.data_names)):
        #     self.data_names[i] = self.data_names[i].split(".")[0]
        #     print(self.data_names[i])
        data_item = tlx.files.load_npy_to_any(self.data_path, self.data_names[index])
        data_item.edge_attr = data_item.edge_attr.transpose(0, 1)
        data_item.edge_type = data_item.edge_type.transpose(0, 1)
        # print("dataset_shape:", data_item.x.shape, data_item.edge_attr.shape)
        return data_item

    def download(self):
        path = download_url(self.url, self.save_to)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            # 解压缩所有文件
            zip_ref.extractall(self.save_to)

