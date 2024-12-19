import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import tensorlayerx as tlx
import tensorlayerx.nn as nn
from gammagl.utils import  to_scipy_sparse_matrix
from gammagl.mpops import *


from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()


        
class MM_Model(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, user_init_embedding, item_attribute_dict):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        self.image_trans = nn.Linear(in_features=image_feats.shape[1], out_features = args.embed_size)
        self.text_trans = nn.Linear(in_features = text_feats.shape[1], out_features = args.embed_size)
        self.user_trans = nn.Linear(in_features = user_init_embedding.shape[1], out_features = args.embed_size)  
        self.item_trans = nn.Linear(in_features = item_attribute_dict['title'].shape[1], out_features = args.embed_size)
        
        init = tlx.initializers.xavier_uniform()
        self.image_trans.weights = nn.Parameter(init(shape = (args.embed_size,image_feats.shape[1]), dtype=tlx.float32))
        self.text_trans.weights = nn.Parameter(init(shape = (args.embed_size,text_feats.shape[1]), dtype=tlx.float32))
        self.user_trans.weights = nn.Parameter(init(shape = (args.embed_size,user_init_embedding.shape[1]), dtype=tlx.float32))
        self.item_trans.weights = nn.Parameter(init(shape = (args.embed_size,item_attribute_dict['title'].shape[1]), dtype=tlx.float32))

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        
        self.user_id_embedding.embeddings=nn.Parameter(init(shape = (n_users,self.embedding_dim), dtype=tlx.float32))
        self.item_id_embedding.embeddings=nn.Parameter(init(shape = (n_items,self.embedding_dim), dtype=tlx.float32))
        
        self.train_weights = []
        self.train_weights.append(self.image_trans.weights)
        self.train_weights.append(self.text_trans.weights)
        self.train_weights.append(self.user_trans.weights)
        self.train_weights.append(self.item_trans.weights)
        self.train_weights.append(self.user_id_embedding.embeddings)
        self.train_weights.append(self.item_id_embedding.embeddings)


        
        self.image_feats = tlx.convert_to_tensor(image_feats,dtype=tlx.float32)
        self.text_feats = tlx.convert_to_tensor(text_feats,dtype=tlx.float32)
        self.user_feats = tlx.convert_to_tensor(user_init_embedding,dtype=tlx.float32)
        self.item_feats = {}
        for key in item_attribute_dict.keys():                                   
            self.item_feats[key] =tlx.convert_to_tensor(item_attribute_dict[key],dtype=tlx.float32)

        self.softmax = nn.Softmax(axis=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(num_features=args.embed_size)
        self.tau = 0.5

    def mm(self, x, y):
        return tlx.matmul(x, y)
    
    def sim(self, z1, z2):
        z1 = tlx.l2_normalize(z1)
        z2 = tlx.l2_normalize(z2)
        return tlx.matmul(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: tlx.exp(x / self.tau)
        indices = tlx.ops.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  
            between_sim = f(self.sim(z1[mask], z2))  

            losses.append(-tlx.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag()+1e-8)))

        loss_vec = tlx.concat(losses)
        return tlx.reduce_mean(loss_vec)

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat
    
    def matrix_to_tensor(self, cur_matrix):
        """
        将稀疏矩阵转换为 TensorLayerX 张量。
        支持 scipy 的 COO 格式或其他格式。
    
        参数:
        cur_matrix: scipy.sparse 矩阵，支持 COO、CSR 或 CSC 格式。
        
        返回:
        tensor_matrix: TensorLayerX 的稀疏张量。
        """
    # 检查输入类型并确保是稀疏矩阵
        if not sp.issparse(cur_matrix):
            raise TypeError("输入必须是 scipy.sparse 矩阵格式")
    
    # 如果不是 COO 格式，则转换为 COO 格式
        if not isinstance(cur_matrix, sp.coo_matrix):
            cur_matrix = cur_matrix.tocoo()
    
    # 转换为 TensorLayerX 的张量
        indices=tlx.convert_to_tensor([cur_matrix.row, cur_matrix.col], dtype=tlx.int64),  # 边索引
        values=tlx.convert_to_tensor(cur_matrix.data, dtype=tlx.float32),                 # 非零值
        size=cur_matrix.shape # 矩阵形状
    # 创建全零张量
        dense_tensor = tlx.zeros(size, dtype=values.dtype)
        for i in range(values.shape[0]):
            row, col = indices[:, i]  # 获取当前非零元素的位置
            dense_tensor[row, col] = values[i]  # 填充非零值

        return dense_tensor
    

    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = tlx.stack(tensors, axis=0)

        return tensors


    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):


        # feature mask 
        i_mask_nodes, u_mask_nodes = None, None
        if args.mask:
            # 生成从 0 到 self.n_items-1 的随机排列
            i_perm = tlx.convert_to_tensor(np.random.permutation(self.n_items))
            i_num_mask_nodes = int(args.mask_rate * self.n_items)
            i_mask_nodes = i_perm[: i_num_mask_nodes]
            for key in self.item_feats.keys():
                self.item_feats[key][i_mask_nodes] = tlx.reduce_mean(self.item_feats[key],axis=0)

        u_perm = tlx.convert_to_tensor(np.random.permutation(self.n_users))
        if args.mask_rate>0:
            u_num_mask_nodes = int(args.mask_rate * self.n_users)
            u_mask_nodes = u_perm[: u_num_mask_nodes]
            self.user_feats[u_mask_nodes] = tlx.reduce_mean(self.user_feats,axis=0)
        
        image_feats = self.image_feats
        image_tmp = self.image_trans(image_feats)
        image_feats = self.dropout(image_tmp)
        
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(tlx.convert_to_tensor(self.user_feats,dtype=tlx.float32)))
        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))

        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # aug item attribute
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # aug user profile
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph, item_prof_feat)

        u_g_embeddings = self.user_id_embedding.embeddings
        i_g_embeddings = self.item_id_embedding.embeddings           

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax( tlx.matmul(ui_graph, i_g_embeddings) ) 
                i_g_embeddings = self.softmax( tlx.matmul(iu_graph, u_g_embeddings) )
            else:
                u_g_embeddings = tlx.matmul(ui_graph, i_g_embeddings) 
                i_g_embeddings = tlx.matmul(iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = tlx.reduce_mean(tlx.stack(user_emb_list), axis=0)
        i_g_embeddings = tlx.reduce_mean(tlx.stack(item_emb_list), axis=0)



        u_g_embeddings =  args.model_cat_rate*tlx.l2_normalize(image_user_feats,axis=1) + args.model_cat_rate*tlx.l2_normalize(text_user_feats,axis=1)
        i_g_embeddings =  i_g_embeddings + args.model_cat_rate*tlx.l2_normalize(image_item_feats,axis=1) + args.model_cat_rate*tlx.l2_normalize(text_item_feats,axis=1)
        # profile
        u_g_embeddings += args.user_cat_rate*tlx.l2_normalize(user_prof_feat,axis=1)
        i_g_embeddings += args.user_cat_rate*tlx.l2_normalize(item_prof_feat,axis=1)

        # attribute 
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate*tlx.l2_normalize(user_feat_from_item[key],axis=1)
            i_g_embeddings += args.item_cat_rate*tlx.l2_normalize(item_feats[key],axis=1) 

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, i_mask_nodes, u_mask_nodes



class Decoder(nn.Module):
    def __init__(self, feat_size):
        super(Decoder, self).__init__()
        self.feat_size=feat_size

        self.u_net = nn.Sequential(
            nn.Linear(args.embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

        self.i_net = nn.Sequential(
            nn.Linear(args.embed_size, int(self.feat_size)),
            nn.LeakyReLU(True),
        )

    def forward(self, u, i):
        u_output = self.u_net(tlx.convert_to_numpy(u,dtype=tlx.float32)) 
        tensor_list = []
        for index,value in enumerate(i.keys()):  
            tensor_list.append(i[value]) 
        i_tensor = tlx.stack(tensor_list)
        i_output = self.i_net(tlx.convert_to_numpy(i_tensor,dtype=tlx.float32))
        return u_output, i_output  


