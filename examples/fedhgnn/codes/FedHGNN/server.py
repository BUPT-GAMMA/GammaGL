import tensorlayerx as tlx
from tensorlayerx import nn
import torch
import os
os.environ['TL_BACKEND'] = 'torch'

from gammagl.models.fedhgnn import *
from eval import *
from util import *

class Server(nn.Module):
    def __init__(self, client_list, model, hg, features, args):
        super().__init__()
        device = args.device
        if 'cuda' in device:
            self.device, self.device_id = device.split(':')[0], device.split(':')[1]
            tlx.set_device(device=self.device, id=self.device_id)
        self.user_emb = nn.Embedding(features[0].shape[0], features[0].shape[1]).cuda()#.to(self.device)
        self.item_emb = nn.Embedding(features[1].shape[0], features[1].shape[1]).cuda()#.to(self.device)
        hg_user = hg[0]
        hg_item = hg[1]



        self.hg_user = hg_user
        self.hg_item = hg_item


        self.client_list = client_list
        self.features = features
        self.model_user = model[0]#(0:model_user, 1: model_item)
        self.model_item = model[1]

        #self.user_emb.weight.data = nn.Parameter(tlx.convert_to_tensor(features[0]))#.to(self.device)
        #self.item_emb.weight.data = nn.Parameter(tlx.convert_to_tensor(features[1]))#.to(self.device)
        self.user_emb.embeddings.data = tlx.convert_to_tensor(features[0], dtype = 'float32')
        self.item_emb.embeddings.data = tlx.convert_to_tensor(features[1], dtype = 'float32')
        #self.user_emb.__dict__['_trainable_weights'][0] = tlx.convert_to_tensor(features[0])
        #self.item_emb.__dict__['_trainable_weights'][0] = tlx.convert_to_tensor(features[1])
        #nn.init.normal_(self.item_emb.weight, std=0.01)
        self.lr = args.lr
        self.weight_decay = args.weight_decay



    def aggregate(self, param_list):
        flag = False
        number = 0
        tlx.set_device(device=self.device, id=self.device_id)
        gradient_item = tlx.zeros_like(self.item_emb.embeddings)
        gradient_user = tlx.zeros_like(self.user_emb.embeddings)
        item_count = tlx.zeros((self.item_emb.embeddings.shape[0],)).cuda()
        user_count = tlx.zeros((self.user_emb.embeddings.shape[0],)).cuda()
        for parameter in param_list:
            model_grad_user, model_grad_item = parameter['model']
            item_grad, returned_items = parameter['item']
            user_grad, returned_users = parameter['user']
            num = len(returned_items)
            item_count[returned_items] += 1
            user_count[returned_users] += num

            number += num
            if not flag:
                flag = True
                gradient_model_user = []
                gradient_model_item = []
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user.append(model_grad_user[i]* num)
                for i in range(len(model_grad_item)):
                    gradient_model_item.append(model_grad_item[i]* num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num
                for i in range(len(model_grad_user)):
                    gradient_model_user[i] += model_grad_user[i] * num
                for i in range(len(model_grad_item)):
                    gradient_model_item[i] += model_grad_item[i] * num

        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1
        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)
        for i in range(len(gradient_model_user)):
            gradient_model_user[i] = gradient_model_user[i] / number
        for i in range(len(gradient_model_item)):
            gradient_model_item[i] = gradient_model_item[i] / number


        #更新model参数
        ls_model_param_user = list(self.model_user.parameters())
        ls_model_param_item = list(self.model_item.parameters())
        for i in range(len(ls_model_param_user)):
            ls_model_param_user[i].data = ls_model_param_user[i].data - self.lr * gradient_model_user[i] - self.weight_decay * ls_model_param_user[i].data
        for i in range(len(ls_model_param_item)):
            ls_model_param_item[i].data = ls_model_param_item[i].data - self.lr * gradient_model_item[i] - self.weight_decay * ls_model_param_item[i].data

        # for i in range(len(list(self.model_user.parameters()))):
        #     print(ls_model_param_user[i].data)
        #     break
        #更新item/user参数
        item_index = gradient_item.sum(dim = -1) != 0
        user_index = gradient_user.sum(dim = -1) != 0
        with torch.no_grad():#不加会报错
            self.item_emb.embeddings[item_index] = self.item_emb.embeddings[item_index] -  self.lr * gradient_item[item_index] - self.weight_decay * self.item_emb.embeddings[item_index]
            self.user_emb.embeddings[user_index] = self.user_emb.embeddings[user_index] -  self.lr * gradient_user[user_index] - self.weight_decay * self.user_emb.embeddings[user_index]



    def distribute(self, client_list):
        for client in client_list:
            client.update(self.model_user, self.model_item)


    def predict(self, test_dataloader, epoch):
        hit_at_5 = []
        hit_at_10 = []
        ndcg_at_5 = []
        ndcg_at_10 = []

        self.model_item.eval()
        self.model_user.eval()
        logits_user = self.model_user(self.hg_user, self.user_emb.embeddings)
        logits_item = self.model_item(self.hg_item, self.item_emb.embeddings)
        for u, i, neg_i in test_dataloader: #test_i算上了test_negative, 真实的放在最后一位[99]
            cur_user = logits_user[u]
            cur_item = logits_item[i]
            rating = tlx.reduce_sum(cur_user * cur_item, axis=-1)#当前client user和所有item点乘(include test item)

            for eva_idx, eva in enumerate(rating):
                cur_neg = logits_item[neg_i[eva_idx]]
                cur_rating_neg = tlx.reduce_sum(cur_user[eva_idx] * cur_neg, axis=-1)
                #print(np.shape(cur_rating_neg))
                cur_eva = tlx.concat([cur_rating_neg, tlx.expand_dims(rating[eva_idx], 0)], axis=0)
                #print(np.shape(rating[eva_idx]))
                # print(cur_eva)
                hit_at_5_ = evaluate_recall(cur_eva, [99], 5)#[99]是测试集(ground truth)
                hit_at_10_ = evaluate_recall(cur_eva, [99], 10)
                ndcg_at_5_ = evaluate_ndcg(cur_eva, [99], 5)
                ndcg_at_10_ = evaluate_ndcg(cur_eva, [99], 10)
                #print(hit_at_10_)
                hit_at_5.append(hit_at_5_)
                hit_at_10.append(hit_at_10_)
                ndcg_at_5.append(ndcg_at_5_)
                ndcg_at_10.append(ndcg_at_10_)
        hit_at_5 = np.mean(np.array(hit_at_5)).item()
        hit_at_10 = np.mean(np.array(hit_at_10)).item()
        ndcg_at_5 = np.mean(np.array(ndcg_at_5)).item()
        ndcg_at_10 = np.mean(np.array(ndcg_at_10)).item()

        logging.info('Epoch: %d, hit_at_5 = %.4f, hit_at_10 = %.4f, ndcg_at_5 = %.4f, ndcg_at_10 = %.4f'
              % (epoch, hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10))
        return hit_at_5, hit_at_10, ndcg_at_5, ndcg_at_10










