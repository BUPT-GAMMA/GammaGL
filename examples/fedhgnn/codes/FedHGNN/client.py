import copy
import tensorlayerx as tlx
import os
os.environ['TL_BACKEND'] = 'torch'
import random

from local_differential_privacy_library import *
from util import *
from random import sample
from sklearn.metrics.pairwise import cosine_similarity

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



class Client(tlx.nn.Module):
    def __init__(self, user_id, item_id, args):
        super().__init__()
        self.device = args.device
        self.user_id = user_id
        self.item_id = item_id #list
        #self.semantic_neighbors = semantic_neighbors


    def negative_sample(self, total_item_num):
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in self.item_id:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in self.item_id:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind

    def negative_sample_with_augment(self, total_item_num, sampled_items):
        item_set = self.item_id+sampled_items
        '''生成item负样本集合'''
        #从item列表里随机选取item作为user的负样本
        item_neg_ind = []
        #item_neg_ind和item_id数量一样
        for _ in item_set:
            neg_item = np.random.randint(1, total_item_num)
            while neg_item in item_set:
                neg_item = np.random.randint(1, total_item_num)
            item_neg_ind.append(neg_item)
        '''生成item负样本集合end'''
        return item_neg_ind

    def sample_item_augment(self, item_num):
        ls = [i for i in range(item_num) if i not in self.item_id]
        sampled_items = sample(ls, 5)

        return sampled_items


    def perturb_adj(self, value, label_author, author_label, label_count, shared_knowledge_rep, eps1, eps2):
        #print(value.shape) #1,17431
        #此用户的item共可分成多少个groups
        groups = {}
        for item in self.item_id:
            group = author_label[item]
            if(group not in groups.keys()):
                groups[group] = [item]
            else:
                groups[group].append(item)

        '''step1:EM'''
        num_groups = len(groups)
        quality = np.array([0.0]*len(label_author))
        G_s_u =  groups.keys()
        if(len(G_s_u)==0):#此用户没有交互的item，则各个位置quality平均
            for group in label_author.keys():
                quality[group] = 1
            num_groups = 1
        else:
            for group in label_author.keys():
                qua = max([(cosine_similarity(shared_knowledge_rep[g], shared_knowledge_rep[group])+1)/2.0 for g in G_s_u])
                quality[group] = qua

        EM_eps = eps1/num_groups
        EM_p = EM_eps*quality/2 #隐私预算1 eps
        EM_p = softmax(EM_p)

        #按照概率选择group
        select_group_keys = np.random.choice(range(len(label_author)), size = len(groups), replace = False, p = EM_p)
        select_group_keys_temp = list(select_group_keys)
        degree_list = [len(v) for _, v in groups.items()]
        new_groups = {}

        for key in select_group_keys:#先把存在于当前用户的shared knowledge拿出来
            key_temp = key
            if(key_temp in groups.keys()):
                new_groups[key_temp] = groups[key_temp]
                degree_list.remove(len(groups[key_temp]))
                select_group_keys_temp.remove(key_temp)

        for key in select_group_keys_temp:#不存在的随机采样交互的item，并保持度一致
            key_temp = key
            cur_degree = degree_list[0]
            if(len(label_author[key_temp]) >= cur_degree):
                new_groups[key_temp] = random.sample(label_author[key_temp], cur_degree)
            else:#需要的度比当前group的size大，则将度设为当前group的size
                new_groups[key_temp] = label_author[key_temp]
            degree_list.remove(cur_degree)

        groups = new_groups
        value = np.zeros_like(value)#一定要更新value
        for group_id, items in groups.items():
            value[:,items] = 1
        '''pure em'''
        #value_rr = value



        '''step2:rr'''
        all_items = set(range(len(author_label)))
        select_items = []
        for group_id, items in groups.items():
            select_items.extend(label_author[group_id])
        mask_rr = list(all_items - set(select_items))

        '''rr'''
        value_rr = perturbation_test(value, 1-value, eps2)
        #print(np.sum(value_rr)) 4648
        value_rr[:, mask_rr] = 0
        # #print(np.sum(value_rr)) 469
        #
        '''dprr'''
        for group_id, items in groups.items():
            degree = len(items)
            n = len(label_author[group_id])
            p = eps2p(eps2)
            q = degree/(degree*(2*p-1) + (n)*(1-p))
            rnd = np.random.random(value_rr.shape)
            #原来是0的一定还是0，原来是1的以概率q保持1，以达到degree减少
            dprr_results = np.where(rnd<q, value_rr, np.zeros((value_rr.shape)))
            value_rr[:, label_author[group_id]] = dprr_results[:, label_author[group_id]]


        #print('....')
        #print(self.item_id)
        #print(value_rr.nonzero()[1])
        return value_rr





    def update(self, model_user, model_item):
        self.model_user = copy.deepcopy(model_user)
        self.model_item = copy.deepcopy(model_item)
        # self.item_emb.weight.data = Parameter(aggr_param['item'].weight.data.clone())


    def train_(self, hg, user_emb, item_emb):
        total_item_num = item_emb.embeddings.data.shape[0]#item_emb.weight.shape[0]
        #user_emb = torch.clone(user_emb.weight).detach()
        user_emb = tlx.identity(user_emb.embeddings.data).detach()
        item_emb = tlx.identity(item_emb.embeddings.data).detach()
        user_emb.requires_grad = True
        item_emb.requires_grad = True
        user_emb.grad = tlx.zeros_like(user_emb)
        item_emb.grad = tlx.zeros_like(item_emb)
        hg_user = hg[0]
        hg_item = hg[1]

        self.model_user.train()
        self.model_item.train()

        #sample_item_augment
        sampled_item = self.sample_item_augment(total_item_num)
        item_neg_id = self.negative_sample_with_augment(total_item_num, sampled_item)
        #item_neg_id = self.negative_sample(total_item_num)
        
        logits_user = self.model_user(hg_user, user_emb)#+user_emb
        logits_item = self.model_item(hg_item, item_emb)#+item_emb

        cur_user = logits_user[self.user_id]
        #cur_item_pos = logits_item[self.item_id]
        cur_item_pos = logits_item[self.item_id+sampled_item]
        cur_item_neg = logits_item[item_neg_id]

        #pos_scores = torch.sum(cur_user * cur_item_pos, dim=-1)
        #neg_scores = torch.sum(cur_user * cur_item_neg, dim=-1)
        pos_scores = tlx.reduce_sum(cur_user * cur_item_pos, axis=-1)
        neg_scores = tlx.reduce_sum(cur_user * cur_item_neg, axis=-1)
        loss = -(pos_scores - neg_scores).sigmoid().log().sum()


        self.model_user.zero_grad()
        self.model_item.zero_grad()

        loss.backward()
        #self.optimizer.step()

        #grad
        model_grad_user = []
        model_grad_item = []
        for param in list(self.model_user.parameters()):
            grad = param.grad#
            model_grad_user.append(grad)
        for param in list(self.model_item.parameters()):
            grad = param.grad#
            model_grad_item.append(grad)

        mask_item = item_emb.grad.sum(-1)!=0#直接通过grad！=0
        updated_items = np.array(range(item_emb.shape[0]))[mask_item.cpu()]#list(set(self.item_id + item_neg_id))
        #print(updated_items)
        item_grad = item_emb.grad[updated_items, :]#


        mask_user = user_emb.grad.sum(-1)!=0
        updated_users = np.array(range(user_emb.shape[0]))[mask_user.cpu()]#list(set([self.user_id] + self.semantic_neighbors))
        #print(len(updated_users))
        user_grad = user_emb.grad[updated_users, :]#
        #print(user_grad)
        # torch.cuda.empty_cache()


        return {'user': (user_grad, updated_users), 'item' : (item_grad, updated_items), 'model': (model_grad_user, model_grad_item)}, \
               loss.detach()
