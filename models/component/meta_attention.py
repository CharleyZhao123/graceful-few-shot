from models import model_register
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys
sys.path.append('..')

@model_register('meta-attention')
class MetaAttention(nn.Module):
    '''
    Task-Level Attention for Meta Learning
    输入: Q: [1, Q, dim], K: [1, W, S, dim]
    输出: O: [1, Q, W, dim]
    '''

    def __init__(self, dim=512, use_scaling=False, similarity_method='cos',
                 nor_type='softmax', way_num=5, shot_num=5, trans_type='adapter', **kargs):
        super().__init__()
        self.dim = dim
        self.use_scaling = use_scaling
        self.scaling = torch.sqrt(torch.tensor(dim).float())
        self.similarity_method = similarity_method
        self.nor_type = nor_type
        self.way_num = way_num
        self.trans_type = 'adapter'

        # 包含了所有需要被优化的参数
        self.vars = nn.ParameterList()

        # Task Gate
        task_gate_indim = self.dim * (1 + shot_num)
        task_gate_weight = nn.Parameter(torch.ones(shot_num, task_gate_indim))
        torch.nn.init.kaiming_normal_(task_gate_weight)
        task_gate_bias = nn.Parameter(torch.zeros(shot_num))

        # Sample-Level Transform
        if trans_type == 'adapter':
            hid_dim = dim//2
            down_linear_weight = nn.Parameter(torch.ones(1, way_num, shot_num, dim, hid_dim))
            torch.nn.init.kaiming_normal_(down_linear_weight)
            up_linear_weight = nn.Parameter(torch.ones(1, way_num, shot_num, hid_dim, dim))
            torch.nn.init.kaiming_normal_(up_linear_weight)            
            self.vars.extend([down_linear_weight, up_linear_weight, task_gate_weight, task_gate_bias])
            # self.vars.extend([down_linear_weight, up_linear_weight])
        else:
            key_trans_weight = nn.Parameter(torch.ones(1, way_num, shot_num, dim, dim))
            self.vars.extend([key_trans_weight, task_gate_weight, task_gate_bias])

    def forward(self, query, key, params=None):
        # ===== 准备 =====
        query_num = query.shape[1]
        way_num = key.shape[1]

        if params is None:
            params = self.vars
        
        # ===== 线性变换 =====
        if self.trans_type == 'adapter':
            down_linear_weight = params[0]  # [1, W, S, dim, hid_dim]
            up_linear_weight = params[1]  # [1, W, S, hid_dim, dim]
            task_gate_weight = params[2]
            task_gate_bias = params[3]
        else:
            key_trans_weight = params[0]
            task_gate_weight = params[1]
            task_gate_bias = params[2]

        new_query = query  # [1, Q, dim]

        if self.trans_type == 'adapter':
            # key: [1, W, S, dim]
            new_key = key.unsqueeze(-2)  # [1, W, S, 1, dim]
            # down linear
            new_key = torch.matmul(new_key, down_linear_weight)  # [1, W, S, 1, hid_dim]

            # relu
            new_key = F.relu(new_key, inplace = [True])

            # up linear 
            new_key = torch.matmul(new_key, up_linear_weight)  # [1, W, S, 1, dim]
            new_key = new_key.squeeze(-2)  # [1, W, S, dim]

            # shortcut connection
            new_value = new_key + key
        else:
            # key: [1, W, S, dim], weight: [1, W, S, dim, dim], new_key: [1, W, S, dim]
            new_key = key.unsqueeze(-2)  # [1, W, S, 1, dim]
            new_key = torch.matmul(new_key, key_trans_weight).squeeze(-2)

            new_value = new_key  # [1, W, S, dim]

        # ===== 得到prototype特征 =====
        # 整理tensor便于计算
        new_key = new_key.unsqueeze(1).repeat(
            1, query_num, 1, 1, 1)  # [1, Q, W, S, dim]
        new_value = new_value.unsqueeze(1).repeat(
            1, query_num, 1, 1, 1)  # [1, Q, W, S, dim]
        new_query = new_query.unsqueeze(2).repeat(
            1, 1, way_num, 1).unsqueeze(-2)  # [1, Q, W, 1, dim]

        # gate过滤: concat & MLP
        gate_input = torch.cat([new_query, new_key], dim=-2)  # [1, Q, W, (1+S), dim]
        gate_input = gate_input.view(query_num*way_num, -1)  # [Q*W, (1+S)*dim]
        gate_output = F.linear(gate_input, weight=task_gate_weight, bias=task_gate_bias)  # [Q*W, S]
        sim = gate_output.view(1, query_num, way_num, -1).unsqueeze(-2)  # [1, Q, W, 1, S]

        # 处理相似度
        if self.use_scaling:
            sim = sim / self.scaling
        if self.nor_type == 'softmax':
            sim = F.softmax(sim, dim=-1)
        elif self.nor_type == 'l2_norm':
            sim = F.normalize(sim, dim=-1)
        # print("1:")
        # print(sim[0, 0, 0, 0, :])

        # 相似度过滤
        if self.similarity_method == 'cos':
            nor_query = F.normalize(new_query, dim=-1)
            nor_key = F.normalize(new_key, dim=-1)

            sim = sim + torch.matmul(nor_query, nor_key.permute(
                0, 1, 2, 4, 3))  # [T, Q, W, 1, S]
            # print("2:")
            # print(sim[0, 0, 0, 0, :])
            sim = F.normalize(sim, dim=-1)
            # print("3:")
            # print(sim[0, 0, 0, 0, :])

        # 加权(相似度)求和
        proto_feat = torch.matmul(sim, new_value).squeeze(-2)  # [T, Q, W, dim]

        return proto_feat

if __name__ == '__main__':
    query=torch.rand((4, 75, 512))
    key=torch.rand((4, 5, 5, 512))
