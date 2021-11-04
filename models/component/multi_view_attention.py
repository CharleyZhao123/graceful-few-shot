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
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from models import model_register

@model_register('dot-attention')
class DotAttention(nn.Module):
    '''
    无参点积Attention
    '''
    def __init__(self, dim=512, use_scaling=False, similarity_method='cos', nor_type='l2_norm',**kargs):
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
        key = key.unsqueeze(1).repeat(1, query_num, 1, 1, 1)  # [T, Q, W, S, dim]
        query = query.unsqueeze(2).repeat(1, 1, way_num, 1).unsqueeze(-2)  # [T, Q, W, 1, dim]

        # 计算Query与Key的相似度
        if self.similarity_method == 'cos':
            nor_query = F.normalize(query, dim=-1)
            nor_key = F.normalize(key, dim=-1)
            sim = torch.matmul(nor_query, nor_key.permute(0, 1, 2, 4, 3))  # [T, Q, W, 1, S]
        else:
            sim = torch.matmul(query, key.permute(0, 1, 2, 4, 3))  # [T, Q, W, 1, S]

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

if __name__ == '__main__':
    query = torch.rand((4, 75, 512))
    key = torch.rand((4, 5, 5, 512))
    dot_attention = DotAttention()
    output = dot_attention(query, key)
    print(output.shape)