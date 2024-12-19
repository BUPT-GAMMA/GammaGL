from datetime import datetime
import math
import os
import random
import sys
from time import time
from tqdm import tqdm
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import tensorlayerx as tlx
import tensorlayerx.nn as nn
import tensorlayerx.optimizers as optim
import random


import copy

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorlayerx.backend import BACKEND
BACKEND = 'tensorflow'


print("TL_BACKEND:", os.environ.get("TL_BACKEND"))

#因为tensorflow-gpu环境不好装
if tlx.BACKEND == 'tensorflow' or 'paddle': tlx.set_device('CPU')
else : tlx.set_device('GPU') 





from utility.parser import parse_args
from Models import MM_Model, Decoder  
from utility.batch_test import *
from utility.logging import Logger
from utility.norm import build_sim, build_knn_normalized_graph

import setproctitle

args = parse_args()


        
class Trainer(object):
    def __init__(self, data_config):
       
        self.task_name = "%s_%s_%s" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.dataset, args.cf_model,)
        self.logger = Logger(filename=self.task_name, is_debug=args.debug)
        self.logger.logging("PID: %d" % os.getpid())
        self.logger.logging(str(args))

        self.mess_dropout = eval(args.mess_dropout)
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.weight_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
 
        self.image_feats = np.load(args.data_path + '{}/image_feat.npy'.format(args.dataset))
        self.text_feats = np.load(args.data_path + '{}/text_feat.npy'.format(args.dataset))
        self.image_feat_dim = self.image_feats.shape[-1]
        self.text_feat_dim = self.text_feats.shape[-1]

        self.ui_graph = self.ui_graph_raw = pickle.load(open(args.data_path + args.dataset + '/train_mat','rb'))
        # get user embedding  
        augmented_user_init_embedding = pickle.load(open(args.data_path + args.dataset + '/augmented_user_init_embedding','rb'))
        augmented_user_init_embedding_list = []
        for i in range(len(augmented_user_init_embedding)):
            augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
        augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
        pickle.dump(augmented_user_init_embedding_final, open(args.data_path + args.dataset + '/augmented_user_init_embedding_final','wb'))
        self.user_init_embedding = pickle.load(open(args.data_path + args.dataset + '/augmented_user_init_embedding_final','rb'))
        # get separate embedding matrix 
        if args.dataset=='preprocessed_raw_MovieLens':
            augmented_total_embed_dict = {'title':[] , 'genre':[], 'director':[], 'country':[], 'language':[]}   
        elif args.dataset=='netflix':
            augmented_total_embed_dict = {'year':[] , 'title':[], 'director':[], 'country':[], 'language':[]}   
        augmented_atttribute_embedding_dict = pickle.load(open(args.data_path + args.dataset + '/augmented_atttribute_embedding_dict','rb'))
        for value in augmented_atttribute_embedding_dict.keys():
            for i in range(len(augmented_atttribute_embedding_dict[value])):
                augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])   
            augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])    
        pickle.dump(augmented_total_embed_dict, open(args.data_path + args.dataset + '/augmented_total_embed_dict','wb'))
        self.item_attribute_embedding = pickle.load(open(args.data_path + args.dataset + '/augmented_total_embed_dict','rb'))       

        self.image_ui_index = {'x':[], 'y':[]}
        self.text_ui_index = {'x':[], 'y':[]}

        self.n_users = self.ui_graph.shape[0]
        self.n_items = self.ui_graph.shape[1]        
        self.iu_graph = self.ui_graph.T

        self.ui_graph = self.coo_norm(self.ui_graph, mean_flag=True)
        self.iu_graph = self.coo_norm(self.iu_graph, mean_flag=True)

        self.ui_graph = self.matrix_to_tensor(self.ui_graph)
        self.iu_graph = self.matrix_to_tensor(self.iu_graph)
        self.image_ui_graph = self.text_ui_graph = self.ui_graph
        self.image_iu_graph = self.text_iu_graph = self.iu_graph

        self.model_mm = MM_Model(self.n_users, self.n_items, self.emb_dim, self.weight_size, self.mess_dropout, self.image_feats, self.text_feats, self.user_init_embedding, self.item_attribute_embedding)      
        self.model_mm = self.model_mm  
        self.decoder = Decoder(self.user_init_embedding.shape[1])


        self.optimizer = optim.Adam( lr=self.lr,grad_clip= tlx.ops.ClipGradByNorm(1.0))  
        
        self.de_optimizer = optim.Adam( lr=args.de_lr)  



    def coo_norm(self, coo_mat, mean_flag=False):
        rowsum = np.array(coo_mat.sum(axis=1) )
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)
        colsum = np.array(coo_mat.sum(axis=0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)
        if mean_flag == False:
            return rowsum_diag*coo_mat*colsum_diag
        else:
            return rowsum_diag*coo_mat
        
        
    def matrix_to_tensor(self, cur_matrix):
        """
        将稀疏矩阵转换为 TensorLayerX 张量。
        支持 scipy 的 COO 格式或其他格式。
    
        参数:
        cur_matrix: scipy.sparse 矩阵，支持 COO、CSR 或 CSC 格式。
        
        返回:
        tensor_matrix: TensorLayerX 的张量。
        """
    # 检查输入类型并确保是稀疏矩阵
        if not sp.issparse(cur_matrix):
            raise TypeError("输入必须是 scipy.sparse 矩阵格式")
    
    # 如果不是 COO 格式，则转换为 COO 格式
        if not isinstance(cur_matrix, sp.coo_matrix):
            cur_matrix = cur_matrix.tocoo()
    
        dense_matrix = cur_matrix.toarray()  # 稀疏矩阵转为稠密矩阵 (NumPy 格式)
        # 转换为 TensorLayerX 的张量
        dense_tensor = tlx.convert_to_tensor(dense_matrix, dtype=tlx.float32)

        return dense_tensor
    

    
    

    def innerProduct(self, u_pos, i_pos, u_neg, j_neg):  
        # 计算 pred_i
        pred_i = tlx.reduce_sum(tlx.multiply(u_pos, i_pos), axis=-1)
        # 计算 pred_j
        pred_j = tlx.reduce_sum(tlx.multiply(u_neg, j_neg), axis=-1)
        return pred_i, pred_j

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0)

    def sim(self, z1, z2):
        z1 = tlx.l2_normalize(z1)
        z2 = tlx.l2_normalize(z2)
        return tlx.matmul(z1, z2,transpose_b=True)

    def feat_reg_loss_calculation(self, g_item_image, g_item_text, g_user_image, g_user_text):
        feat_reg = 1./2*tlx.reduce_sum(g_item_image**2+1e-8) + 1./2*tlx.reduce_sum(g_item_text**2+1e-8) \
            + 1./2*tlx.reduce_sum(g_user_image**2+1e-8) + 1./2*tlx.reduce_sum(g_user_text**2+1e-8)       
        feat_reg = feat_reg / self.n_items
        feat_emb_loss = args.feat_reg_decay * feat_reg
        return feat_emb_loss

    def prune_loss(self, pred, drop_rate):
        ind_sorted = tlx.argsort(pred, axis=0, descending=False)
        loss_sorted = tlx.gather(pred,ind_sorted)
        remember_rate = 1 - drop_rate
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss_update = tlx.gather(pred,ind_update)
        return tlx.reduce_mean(loss_update)

    def mse_criterion(self, x, y, alpha=3):
        x = tlx.l2_normalize(x,axis=-1)
        y = tlx.l2_normalize(y, axis=-1)
        tmp_loss = (1 - tlx.reduce_sum(x * y,axis=-1)).pow_(alpha)
        tmp_loss = tlx.reduce_mean(tmp_loss)
        loss = tlx.losses.mean_squared_error(x, y)
        return loss

    def sce_criterion(self, x, y, alpha=1):
        x = tlx.l2_normalize(x, axis=-1)
        y = tlx.l2_normalize(y, axis=-1)
        loss = (1-tlx.reduce_sum(x*y,axis=-1)).pow_(alpha)
        loss = tlx.reduce_mean(loss)
        return loss

    def test(self, users_to_test, is_val):
        self.model_mm.set_eval()
        ua_embeddings, ia_embeddings, *rest = self.model_mm(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
        result = test_model(ua_embeddings, ia_embeddings, users_to_test, is_val)
        return result

    def train(self):

        now_time = datetime.now()
        run_time = datetime.strftime(now_time,'%Y_%m_%d__%H_%M_%S')

        training_time_list = []
        stopping_step = 0
        train_weights = self.model_mm.train_weights
        
        n_batch = data_generator.n_train // args.batch_size + 1
        best_recall = 0
        for epoch in range(args.epoch):
            t1 = time()
            loss, mf_loss, emb_loss, reg_loss = 0., 0., 0., 0.
            contrastive_loss = 0.
            n_batch = data_generator.n_train // args.batch_size + 1
            sample_time = 0.
            build_item_graph = True

            self.gene_u, self.gene_real, self.gene_fake = None, None, {}
            self.topk_p_dict, self.topk_id_dict = {}, {}
            

            for idx in tqdm(range(n_batch)):
                
                
                self.model_mm.set_train()

                sample_t1 = time()
                users, pos_items, neg_items = data_generator.sample()

                # augment samples 
                augmented_sample_dict = pickle.load(open(args.data_path + args.dataset + '/augmented_sample_dict','rb'))
                users_aug = random.sample(users, int(len(users)*args.aug_sample_rate))
                pos_items_aug = [augmented_sample_dict[user][0] for user in users_aug if (augmented_sample_dict[user][0]<self.n_items and augmented_sample_dict[user][1]<self.n_items)]
                neg_items_aug = [augmented_sample_dict[user][1] for user in users_aug if (augmented_sample_dict[user][0]<self.n_items and augmented_sample_dict[user][1]<self.n_items)]
                users_aug = [user for user in users_aug if (augmented_sample_dict[user][0]<self.n_items and augmented_sample_dict[user][1]<self.n_items)]
                self.new_batch_size = len(users_aug)
                users += users_aug
                pos_items += pos_items_aug
                neg_items += neg_items_aug
                pos_items = clean_negative_items(pos_items) 
                neg_items = clean_negative_items(neg_items) 


                sample_time += time() - sample_t1       
                
                if tlx.BACKEND == 'tensorflow':
                    import tensorflow as tf
                    with tf.GradientTape(persistent=True) as tape:
                        batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss=self.compute_loss(pos_items, neg_items, users)
                        loss = batch_loss

                    grad =  tape.gradient(loss, train_weights)
                    if batch_reg_loss==0.0 : grad[4] = 0.0
                    del tape
                    
                else :
                    batch_loss, batch_mf_loss, batch_emb_loss, batch_reg_loss=self.compute_loss(pos_items, neg_items, users)
                    grad = self.optimizer.gradient(loss = batch_loss,weights = train_weights,return_grad=True)

                self.optimizer.apply_gradients(zip(grad,train_weights))
                mf_loss += float(batch_mf_loss)
                emb_loss += float(batch_emb_loss)
                reg_loss += float(batch_reg_loss)
                
    
            

            if math.isnan(loss) == True:
                self.logger.logging('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % args.verbose != 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f  + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss, reg_loss, contrastive_loss)
                training_time_list.append(time() - t1)
                self.logger.logging(perf_str)

            t2 = time()
            users_to_test = list(data_generator.test_set.keys())
            users_to_val = list(data_generator.val_set.keys())
            ret = self.test(users_to_test, is_val=False)  #^-^
            training_time_list.append(t2 - t1)

            t3 = time()

            if args.verbose > 0:
                perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f], recall=[%.5f, %.5f, %.5f, %.5f], ' \
                           'precision=[%.5f, %.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f, %.5f]' % \
                           (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, reg_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2],
                            ret['recall'][-1],
                            ret['precision'][0], ret['precision'][1], ret['precision'][2], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][1], ret['hit_ratio'][2], ret['hit_ratio'][-1],
                            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][-1])
                self.logger.logging(perf_str)

            if ret['recall'][1] > best_recall:
                best_recall = ret['recall'][1]
                test_ret = self.test(users_to_test, is_val=False)
                self.logger.logging("Test_Recall@%d: %.5f,  precision=[%.5f], ndcg=[%.5f]" % (eval(args.Ks)[1], test_ret['recall'][1], test_ret['precision'][1], test_ret['ndcg'][1]))
                stopping_step = 0
            elif stopping_step < args.early_stopping_patience:
                stopping_step += 1
                self.logger.logging('#####Early stopping steps: %d #####' % stopping_step)
            else:
                self.logger.logging('#####Early stop! #####')
                break
        self.logger.logging(str(test_ret))

        return best_recall, run_time 


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tlx.reduce_sum(tlx.multiply(users, pos_items), axis=1)
        neg_scores = tlx.reduce_sum(tlx.multiply(users, neg_items), axis=1)

        regularizer = 1./(tlx.reduce_sum(2*(users**2))+1e-8) + 1./(tlx.reduce_sum(2*(pos_items**2))+1e-8) + 1./(tlx.reduce_sum(2*(neg_items**2))+1e-8)        
        regularizer = regularizer / self.batch_size
        difference = pos_scores - neg_scores

        maxi =safe_log_sigmoid(difference )
        
        mf_loss = - self.prune_loss(maxi, args.prune_loss_drop_rate)

        emb_loss = self.decay * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss
    
    def compute_loss(self,pos_items, neg_items, users):
        user_presentation_h, item_presentation_h, image_i_feat, text_i_feat, image_u_feat, text_u_feat \
                                , user_prof_feat_pre, item_prof_feat_pre, user_prof_feat, item_prof_feat, user_att_feats, item_att_feats, i_mask_nodes, u_mask_nodes \
                        = self.model_mm.forward(self.ui_graph, self.iu_graph, self.image_ui_graph, self.image_iu_graph, self.text_ui_graph, self.text_iu_graph)
                
        users = tlx.convert_to_tensor(users, dtype=tlx.int64)
        u_bpr_emb = tlx.gather(user_presentation_h, users)
        i_bpr_pos_emb = tlx.gather(item_presentation_h,pos_items)

        i_bpr_neg_emb = tlx.gather(item_presentation_h,neg_items)
                
                
        # modal feat
        image_u_bpr_emb = tlx.gather(image_u_feat,users)
        image_i_bpr_pos_emb = tlx.gather(image_i_feat,pos_items)
        image_i_bpr_neg_emb = tlx.gather(image_i_feat,neg_items)
                
        text_u_bpr_emb = tlx.gather(text_u_feat,users)
        text_i_bpr_pos_emb = tlx.gather(text_i_feat,pos_items)
        text_i_bpr_neg_emb = tlx.gather(text_i_feat,neg_items)
                
        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(u_bpr_emb, i_bpr_pos_emb, i_bpr_neg_emb)
        image_batch_mf_loss, image_batch_emb_loss, image_batch_reg_loss = self.bpr_loss(image_u_bpr_emb, image_i_bpr_pos_emb, image_i_bpr_neg_emb)
        text_batch_mf_loss, text_batch_emb_loss, text_batch_reg_loss = self.bpr_loss(text_u_bpr_emb, text_i_bpr_pos_emb, text_i_bpr_neg_emb)
        mm_mf_loss = image_batch_mf_loss + text_batch_mf_loss
        batch_mf_loss_aug = 0 
        for index,value in enumerate(item_att_feats):  # 
            u_g_embeddings_aug = tlx.gather(user_prof_feat,users)
            pos_i_g_embeddings_aug = tlx.gather(item_att_feats[value],pos_items)
            neg_i_g_embeddings_aug = tlx.gather(item_att_feats[value],neg_items)
            tmp_batch_mf_loss_aug, batch_emb_loss_aug, batch_reg_loss_aug = self.bpr_loss(u_g_embeddings_aug, pos_i_g_embeddings_aug, neg_i_g_embeddings_aug)
            batch_mf_loss_aug += tmp_batch_mf_loss_aug

        feat_emb_loss = self.feat_reg_loss_calculation(image_i_feat, text_i_feat, image_u_feat, text_u_feat)

        att_re_loss = 0
        if args.mask:
            input_i = {} 
            for index,value in enumerate(item_att_feats.keys()):  
                input_i[value] = item_att_feats[value][i_mask_nodes]
            decoded_u, decoded_i = self.decoder(tlx.convert_to_tensor(user_prof_feat[u_mask_nodes]), input_i)
            if args.feat_loss_type=='mse':
                att_re_loss += self.mse_criterion(decoded_u, tlx.convert_to_tensor(self.user_init_embedding[u_mask_nodes]), alpha=args.alpha_l)
                for index,value in enumerate(item_att_feats.keys()):  
                    att_re_loss += self.mse_criterion(decoded_i[index], tlx.convert_to_tensor(self.item_attribute_embedding[value][i_mask_nodes]), alpha=args.alpha_l)
            elif args.feat_loss_type=='sce':
                att_re_loss += self.sce_criterion(decoded_u, tlx.convert_to_tensor(self.user_init_embedding[u_mask_nodes]), alpha=args.alpha_l) 
                for index,value in enumerate(item_att_feats.keys()):  
                    att_re_loss += self.sce_criterion(decoded_i[index], tlx.convert_to_tensor(self.item_attribute_embedding[value][i_mask_nodes]), alpha=args.alpha_l)
        #+ ssl_loss2 #+ batch_contrastive_loss
        batch_loss = batch_mf_loss + batch_emb_loss + batch_reg_loss + feat_emb_loss + args.aug_mf_rate*batch_mf_loss_aug + args.mm_mf_rate*mm_mf_loss + args.att_re_rate*att_re_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

        return batch_loss, batch_mf_loss,batch_emb_loss, batch_reg_loss
    
    
    

    
def safe_log_sigmoid(x):
    # 限制 x 的值在合理范围内
        clipped_x = tlx.clip_by_value(x, -70.0, 70.0)
        return tlx.log(1 / (1 + tlx.exp(-clipped_x)))
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tlx.set_seed(seed) 

def clean_negative_items(items):
    return [item if item >= 0 else 0 for item in items]



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    trainer = Trainer(data_config=config)
    trainer.train()






