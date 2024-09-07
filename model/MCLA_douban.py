import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
import torch.nn.functional as F
from util.conf import OptionConf
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import os
from data.ui_graph import Rating_Graph
from data.social import Relation
import numpy as np

class MCLA_douban(GraphRecommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(MCLA_douban, self).__init__(conf, training_set, test_set, **kwargs)
        args = OptionConf(self.config['MCLA_douban'])
        self.social_layers = int(args['-social_layer'])
        self.rating_layers = int(args['-rating_layer'])
        self.cl_temp1 = float(args['-temp1'])
        self.emb_comb = float(args['-emb_comb'])
        self.ss_rate1 = float(args['-ss_rate1'])
        self.eps_up = float(args['-eps_up'])
        self.eps_down = float(args['-eps_down'])
        self.rating_emb_size = int(self.config['rating.embedding.size'])
        self.rating_split_data = {}
        for each_rating in kwargs['rating.split']:
            self.rating_split_data[int(each_rating)] = Rating_Graph(conf, kwargs['rating.split'][each_rating], self.data.user, self.data.item, self.data.id2user, self.data.id2item)
        self.friend_data = Relation(conf, kwargs['friend.data'], self.data.user)
        self.group_data = Relation(conf, kwargs['group.data'], self.data.user)
        self.model = MCLA_douban_Encoder(self.data, self.friend_data, self.group_data, self.emb_size, self.rating_emb_size,
                                  self.social_layers, self.rating_layers, self.eps_up, self.eps_down, self.emb_comb, self.rating_split_data)

    def print_model_info(self):
        super(MCLA_douban, self).print_model_info()
        print('Friend data size: (user number: %d, relation number: %d).' % (self.friend_data.size()))
        print('Group data size: (user number: %d, relation number: %d).' % (self.group_data.size()))
        print('=' * 80)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch

                user_comb_scores, item_comb_scores, social_comb_scores, rec_user_emb, rec_item_emb, \
                rating_user_embeddings_dic, rating_item_embeddings_dic, social_user_embeddings_dic, \
                mixed_rating_user_embs, mixed_social_embs = model()

                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                self_cl_loss = self.cal_self_cl_loss(user_idx, pos_idx) * self.ss_rate1
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                reg_loss = l2_reg_loss(self.reg, user_emb, pos_item_emb, neg_item_emb) / self.batch_size
                batch_loss = rec_loss + reg_loss + self_cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 10 == 0 and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), ' rec_loss:', rec_loss.item(), ' reg_loss:', reg_loss.item())
            with torch.no_grad():
                user_comb_scores, item_comb_scores, social_comb_scores, self.user_emb, self.item_emb, \
                self.rating_user_embeddings_dic, self.rating_item_embeddings_dic, self.social_user_embeddings_dic, \
                self.mixed_rating_user_embs, self.mixed_social_embs = model()
                self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_self_cl_loss(self, uid, iid):
        _1, _2, _3, _4, _5, rating_user_view_1, rating_item_view_1, social_view_1, _6, _7 = self.model(perturbed=True)
        __1, __2, __3, __4, __5, rating_user_view_2, rating_item_view_2, social_view_2, __6, __7 = self.model(perturbed=True)
        cl_loss = 0
        for i in [1, 2, 3, 4, 5]:
            user_cl_loss = InfoNCE(rating_user_view_1[i][uid], rating_user_view_2[i][uid], self.cl_temp1)
            item_cl_loss = InfoNCE(rating_item_view_1[i][iid], rating_item_view_2[i][iid], self.cl_temp1)
            cl_loss += user_cl_loss
            cl_loss += item_cl_loss
        return cl_loss

    def save(self):
        with torch.no_grad():
            _, __, ___, self.best_user_emb, self.best_item_emb, \
            self.rating_user_embeddings_dic, self.rating_item_embeddings_dic, self.social_user_embeddings_dic, \
            self.mixed_rating_user_embs, self.mixed_social_embs = self.model.forward()

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()

def min_max_range(x, range_values):
    return [round(((xx - min(x)) / (1.0 * (max(x) - min(x)))) * (range_values[1] - range_values[0]) + range_values[0], 4) for xx in tqdm(x)]


class MCLA_douban_Encoder(nn.Module):
    def __init__(self, data, friend_data, group_data, emb_size, rating_emb_size, social_layers, rating_layers, eps_up, eps_down, emb_comb, rating_split_data):
        super(MCLA_douban_Encoder, self).__init__()
        self.data = data
        self.friend_data = friend_data
        self.group_data = group_data
        self.rating_split_data = rating_split_data
        self.latent_size = emb_size
        self.rating_emb_size = rating_emb_size
        self.social_layers = social_layers
        self.rating_layers = rating_layers
        self.eps_up = eps_up
        self.eps_down = eps_down
        self.emb_comb = emb_comb

        self.norm_adj = data.norm_adj
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

        self.rating_norm_adj = {}
        for each_rating in rating_split_data:
            self.rating_norm_adj[each_rating] = rating_split_data[each_rating].norm_adj

        self.sparse_raring_norm_adj = {}
        for each_rating in self.rating_norm_adj:
            self.sparse_raring_norm_adj[each_rating] = TorchGraphInterface.convert_sparse_mat_to_tensor(self.rating_norm_adj[each_rating]).cuda()

        self.friend_H = TorchGraphInterface.convert_sparse_mat_to_tensor(friend_data.social_norm_mat).cuda()
        self.group_H = TorchGraphInterface.convert_sparse_mat_to_tensor(group_data.social_norm_mat).cuda()

        self.rating_interaction_num = {}
        for each_rating in rating_split_data:
            self.rating_interaction_num[each_rating] = rating_split_data[each_rating].interaction_num_dic
        self.social_interaction_num = {}
        self.social_interaction_num['friend'] = friend_data.social_num_dic
        self.social_interaction_num['group'] = group_data.social_num_dic

        self.rating_noise_vectors = {}
        print("获取各个通道的细粒度对比学习噪音大小...")
        if os.path.isfile('noise\\Music_noise_vectors_' + str(eps_down) + '-' + str(eps_up) + '.pickle'):
            print('用户活跃度文件存在')
            with open('noise\\Music_noise_vectors_' + str(eps_down) + '-' + str(eps_up) + '.pickle', 'rb') as f:
                self.rating_noise_vectors = pickle.load(f)
        else:
            print('用户活跃度文件不存在，正在计算......')
            for each_rating in self.rating_interaction_num:
                interaction_num_user_list = self.rating_interaction_num[each_rating]['user'].values()
                rating_noise_list_user = min_max_range(interaction_num_user_list, (self.eps_down, self.eps_up))
                interaction_num_item_list = self.rating_interaction_num[each_rating]['item'].values()
                rating_noise_list_item = min_max_range(interaction_num_item_list, (self.eps_down, self.eps_up))
                rating_noise_list = rating_noise_list_user + rating_noise_list_item
                self.rating_noise_vectors[each_rating] = rating_noise_list
            with open('noise\\Music_noise_vectors_' + str(eps_down) + '-' + str(eps_up) + '.pickle', 'wb') as f:
                pickle.dump(self.rating_noise_vectors, f)
            print('保存成功！')

        self.all_interaction_num = data.interaction_num_dic
        interaction_num_user_list = list(self.all_interaction_num['user'].values())
        interaction_constant_list = torch.tensor([i for i in interaction_num_user_list])
        log_interaction_constant = torch.log(interaction_constant_list)
        log_interaction_mean = torch.mean(log_interaction_constant)
        log_interaction_ratio = log_interaction_constant / log_interaction_mean

        self.adaptive_emb_comb = log_interaction_ratio

        self.init_emb_SGU, self.rating_embedding_dict, self.social_embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        init_para_SGU_dic = {}
        init_para_SGU_dic['user_att'] = nn.Linear(self.latent_size, 1, bias=False)
        init_para_SGU_dic['user_att_mat'] = nn.Linear(self.latent_size, self.latent_size, bias=False)
        init_para_SGU_dic['item_att'] = nn.Linear(self.latent_size, 1, bias=False)
        init_para_SGU_dic['item_att_mat'] = nn.Linear(self.latent_size, self.latent_size, bias=False)
        init_para_SGU_dic['social_att'] = nn.Linear(self.latent_size, 1, bias=False)
        init_para_SGU_dic['social_att_mat'] = nn.Linear(self.latent_size, self.latent_size, bias=False)
        init_emb_SGU_dic = nn.ParameterDict(init_para_SGU_dic)

        rating_para_dic = {}
        for each_rating in self.rating_split_data:
            rating_para_dic['user_emb_' + str(each_rating)] = nn.Parameter(
                initializer(torch.empty(self.data.user_num, self.rating_emb_size)))
            rating_para_dic['item_emb_' + str(each_rating)] = nn.Parameter(
                initializer(torch.empty(self.data.item_num, self.rating_emb_size)))
        rating_embedding_dict = nn.ParameterDict(rating_para_dic)

        social_para_dic = {}
        social_para_dic['user_emb_friend'] = nn.Parameter(
            initializer(torch.empty(self.data.user_num, self.rating_emb_size)))
        social_para_dic['user_emb_group'] = nn.Parameter(
            initializer(torch.empty(self.data.user_num, self.rating_emb_size)))
        social_embedding_dict = nn.ParameterDict(social_para_dic)
        return init_emb_SGU_dic, rating_embedding_dict, social_embedding_dict, social_emb_filter


    def cal_att_scores_and_embs(self, embeddings_dic, type):
        weights = []
        for rating in sorted(embeddings_dic):
            weights.append(torch.tanh(torch.sum(
                self.init_emb_SGU[str(type) + '_att'](self.init_emb_SGU[str(type) + '_att_mat'](embeddings_dic[rating])), dim=1)))  # 加了Tanh激活函数
        weights_tensor = torch.stack(weights, dim=0)
        scores = torch.softmax(weights_tensor, dim=0)
        mixed_embeddings = 0
        i = 0
        for rating in sorted(embeddings_dic):
            socre = torch.transpose(scores[i].repeat(self.latent_size, 1), 0, 1)
            mixed_embeddings += torch.mul(embeddings_dic[rating], socre)
            i += 1
        return mixed_embeddings, scores

    def forward(self, perturbed=False):
        rating_user_embeddings_dic = {}
        rating_item_embeddings_dic = {}
        for each_rating in self.rating_split_data:
            if each_rating in [1, 2, 3, 4, 5]:
                ego_rating_embeddings = torch.cat([self.rating_embedding_dict['user_emb_' + str(each_rating)], self.rating_embedding_dict['item_emb_' + str(each_rating)]], 0)
                rating_embeddings = []

                for k in range(self.rating_layers):
                    ego_rating_embeddings = self.sparse_raring_norm_adj[each_rating] @ ego_rating_embeddings
                    if perturbed:
                        random_noise = torch.rand_like(ego_rating_embeddings).cuda()
                        personized_noise = torch.tensor(np.expand_dims(np.array(self.rating_noise_vectors[each_rating]), axis=0).repeat(self.latent_size, axis=0).T).cuda()
                        ego_rating_embeddings += torch.sign(ego_rating_embeddings) * F.normalize(random_noise, dim=-1) * personized_noise
                    rating_embeddings += [ego_rating_embeddings]

                rating_embeddings = torch.stack(rating_embeddings, dim=1)
                rating_embeddings = torch.mean(rating_embeddings, dim=1)
                user_rating_embeddings = rating_embeddings[:self.data.user_num]
                item_rating_embeddings = rating_embeddings[self.data.user_num:]
                rating_user_embeddings_dic[each_rating] = user_rating_embeddings
                rating_item_embeddings_dic[each_rating] = item_rating_embeddings

        mixed_rating_user_embs, user_scores = self.cal_att_scores_and_embs(rating_user_embeddings_dic, type='user')
        mixed_rating_item_embs, item_scores = self.cal_att_scores_and_embs(rating_item_embeddings_dic, type='item')

        social_user_embeddings_dic = {}

        self.social_embedding_dict = self.social_embedding_dict.cpu()
        friend_user_emb = self.social_embedding_dict['user_emb_friend'].cuda()
        group_user_emb = self.social_embedding_dict['user_emb_group'].cuda()
        all_friend_user_embeddings = []
        all_group_user_embeddings = []

        for s in range(self.social_layers):
            friend_user_emb = self.friend_H @ friend_user_emb
            friend_user_emb = F.normalize(friend_user_emb, p=2, dim=1)
            all_friend_user_embeddings += [friend_user_emb]
            group_user_emb = self.group_H @ group_user_emb
            group_user_emb = F.normalize(group_user_emb, p=2, dim=1)
            all_group_user_embeddings += [group_user_emb]

        all_friend_user_embeddings = torch.stack(all_friend_user_embeddings, dim=1)
        friend_user_embeddings = torch.mean(all_friend_user_embeddings, dim=1)
        all_group_user_embeddings = torch.stack(all_group_user_embeddings, dim=1)
        group_user_embeddings = torch.mean(all_group_user_embeddings, dim=1)

        social_user_embeddings_dic[1] = friend_user_embeddings
        social_user_embeddings_dic[2] = group_user_embeddings

        mixed_social_embs, social_scores = self.cal_att_scores_and_embs(social_user_embeddings_dic, type='social')

        emb_sim = torch.cosine_similarity(mixed_rating_user_embs, mixed_social_embs, dim=1)
        emb_sim_up_0 = torch.clamp(emb_sim, min=0.0)
        adaptive_comb_rate = self.emb_comb / ((self.social_layers + self.rating_layers) / 2 + emb_sim_up_0 * self.adaptive_emb_comb.cuda()).unsqueeze(-1).repeat(1, self.latent_size).cuda()
        mixed_user_embs = mixed_rating_user_embs + mixed_social_embs * adaptive_comb_rate
        mixed_item_embs = mixed_rating_item_embs.cuda()
        return user_scores, item_scores, social_scores, mixed_user_embs, mixed_item_embs, \
               rating_user_embeddings_dic, rating_item_embeddings_dic, social_user_embeddings_dic, \
               mixed_rating_user_embs, mixed_social_embs