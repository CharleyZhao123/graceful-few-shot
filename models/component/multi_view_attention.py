'''
可复用多视角注意力模型:
Simple-Query:
    基础形式
    输入: 一组Query向量和一组Key向量. Q: [T, Q, dim], K: [T, W, S, dim];
    输出: 针对Query向量中每个Query的一个Key的聚合. O: [T, Q, W, dim]
    示例: 每个batch有4个task的5-way 5-shot 15-query任务: 
        输入: Q: [4, 5x15, 512], K: [4, 5, 5, 512]
        输出: O: [4, 5x15, 5, 512]
Multi-Query:
    拓展形式, 暂不考虑
'''
# from models import model_register
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.append('..')


# @model_register('dot-attention')
class DotAttention(nn.Module):
    '''
    无参点积Attention
    '''

    def __init__(self, dim=512, use_scaling=False, similarity_method='cos', nor_type='l2_norm', **kargs):
        super().__init__()
        self.dim = dim
        self.use_scaling = use_scaling
        self.scaling = torch.sqrt(torch.tensor(dim).float())
        self.similarity_method = similarity_method
        self.nor_type = nor_type

    def forward(self, query, key):

        # 整理tensor便于计算
        query_num = query.shape[1]
        way_num = key.shape[1]
        key = key.unsqueeze(1).repeat(
            1, query_num, 1, 1, 1)  # [T, Q, W, S, dim]
        query = query.unsqueeze(2).repeat(
            1, 1, way_num, 1).unsqueeze(-2)  # [T, Q, W, 1, dim]

        # 计算Query与Key的相似度
        if self.similarity_method == 'cos':
            nor_query = F.normalize(query, dim=-1)
            nor_key = F.normalize(key, dim=-1)
            sim = torch.matmul(nor_query, nor_key.permute(
                0, 1, 2, 4, 3))  # [T, Q, W, 1, S]
        else:
            sim = torch.matmul(query, key.permute(
                0, 1, 2, 4, 3))  # [T, Q, W, 1, S]

        # 处理相似度
        if self.use_scaling:
            sim = sim / self.scaling
        if self.nor_type == 'softmax':
            sim = F.softmax(sim, dim=-1)
        elif self.nor_type == 'l2_norm':
            sim = F.normalize(sim, dim=-1)
        # print(sim[0, 0, 0, 0, :])

        # 加权(相似度)求和
        output = torch.matmul(sim, key).squeeze(-2)  # [T, Q, W, dim]

        return output


class LinearTrans(nn.Module):
    '''
    线性变换类
    '''
    def __init__(self, dim=512, way_num=5, type='query'):
        super(LinearTrans, self).__init__()
        self.type = type
        # 构建映射矩阵
        if self.type == 'query':
            self.weight = nn.Parameter(torch.eye(dim))  # [dim, dim]
        else:  # key, value
            eye_base = torch.eye(dim)
            eye_repeat = eye_base.unsqueeze(0).repeat(way_num, 1, 1)
            self.weight = nn.Parameter(eye_repeat) # [way_num, dim, dim]

    def forward(self, in_feat):
        if self.type == 'query':
            # [1, Q, dim]
            out_feat = torch.tensordot(in_feat, self.weight, dims=([2], [0]))
        else:
            # [1, W, S, dim]
            out_feat = torch.matmul(in_feat, self.weight.unsqueeze(0))
        return out_feat


# @model_register('w-attention')
class WAttention(nn.Module):
    '''
    含KQV映射矩阵的Attention
    输入: Q: [1, Q, dim], K: [1, W, S, dim]
    '''

    def __init__(self, dim=512, use_scaling=False, similarity_method='cos',
                 nor_type='l2_norm', way_num=15, query_num=15, **kargs):
        super().__init__()
        self.dim = dim
        self.way_num = way_num
        self.query_num = query_num

        # 构建参数矩阵
        self.query_trans = LinearTrans(self.dim, self.way_num, 'query')
        self.key_trans = LinearTrans(self.dim, self.way_num, 'key')
        self.value_trans = LinearTrans(self.dim, self.way_num, 'value')
    
    def feature_augmentation(self, feat, aug_type='none'):
        if aug_type == 'none':
            return feat
    
    def build_fake_trainset(self, key, choice_type='random', choice_num=1, aug_type='none', epoch=0):
        '''
        利用support set数据(key)构建假训练集
        输入: key: [1, W, S, dim]
        输出: 
            fkey: [1, W, S, dim]
            fquery: [1, Q, dim]
            flabel: [1, Q, W]
            Q = W x choice_num
        '''
        # 准备
        way_num = key.shape[1]
        shot_num = key.shape[2]
        dim = key.shape[3]

        fquery = []
        fkey = key.clone().detach()
        flabel = []

        # 生成筛选数据索引
        if choice_type == 'random':
            choice_id_list = []
            for w in range(way_num):
                random.seed(epoch+w)
                w_choice_id_list = random.sample(range(0, shot_num), choice_num)
                choice_id_list.append(w_choice_id_list)  # [[...], [], ...]

        # 根据choice_id_list构建假训练集
        for w_index, w_choice_id_list in enumerate(choice_id_list):
            for s_index in w_choice_id_list:
                # print(w_index, ' ', s_index)
                # 得到用作query的key特征
                query_feat = fkey[0, w_index, s_index, :].clone().detach()
                fquery.append(query_feat)

                # 构建label
                query_label = [0]*way_num
                query_label[w_index] = 1
                flabel.append(torch.tensor(query_label, dtype=torch.int))

                # 增强对应的原key特征
                aug_feat = self.feature_augmentation(query_feat, aug_type)
                fkey[0, w_index, s_index, :] = aug_feat
        
        fquery = torch.stack(fquery, dim=0).unsqueeze(0)
        flabel = torch.stack(flabel, dim=0).unsqueeze(0)

        return fkey, fquery, flabel

    def train_trans_weight(self, key):
        '''
        使用support set数据(key)训练变换模块
        '''
        fkey, fquery, flabel = self.build_fake_trainset(key)

        
        pass

    def forward(self, query, key):
        '''

        '''
        query_num = self.query_num
        way_num = self.way_num

        # 线性变换
        new_query = self.query_trans(query)  # [1, Q, dim]
        new_key = self.key_trans(key)  # [1, W, S, dim]
        new_value = self.value_trans(key)  # [1, W, S, dim]
        
        if query.equal(new_query):
            print('query == new_query')
        if key.equal(new_key):
            print('key == new_key')
        if key.equal(new_value):
            print('key == new_value')
        
        self.train_trans_weight(key)
        return 0


if __name__ == '__main__':
    query = torch.rand((4, 75, 512))
    key = torch.rand((4, 15, 5, 512))

    w_attention = WAttention()
    w_attention(query, key)