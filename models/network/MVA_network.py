import torch
import torch.nn as nn
import torch.nn.functional as F

from models import model_register, build_model
import utils.few_shot as fs


@model_register('mva-network')
class MVANetwork(nn.Module):
    def __init__(self, encoder_name='resnet12', encoder_args={}, encoder_load_para={},
                 mva_name='dot-attention', mva_args={}, mva_load_para={}, task_info={},
                 similarity_method='cos'):
        super().__init__()
        # 子模块构建
        self.encoder = build_model(
            encoder_name, encoder_args, encoder_load_para)
        self.mva = build_model(mva_name, mva_args, mva_load_para)

        # 任务信息获得
        self.batch_size = task_info['batch_size']
        self.shot_num = task_info['shot_num']
        self.way_num = task_info['way_num']
        self.query_num = task_info['query_num']

        # 其他
        self.similarity_method = similarity_method

    def forward(self, image):
        # ===== 提取特征并整理 =====
        feature = self.encoder(image)

        # 划分特征
        # shot_feat: [T, W, S, dim]
        # query_feat: [T, Q, dim]
        shot_feat, query_feat = fs.split_shot_query(
            feature, self.way_num, self.shot_num, self.query_num, self.batch_size)

        # ===== 将特征送入MVA进行计算 =====
        # proto_feat: [T, Q, W, dim]
        proto_feat = self.mva(query_feat, shot_feat)

        # ===== 得到最终的分类logits =====
        if self.similarity_method == 'cos':
            proto_feat = F.normalize(
                proto_feat, dim=-1).permute(0, 1, 3, 2)  # [T, Q, dim, W]
            query_feat = F.normalize(
                query_feat, dim=-1).unsqueeze(-2)  # [T, Q, 1, dim]
            logits = torch.matmul(query_feat, proto_feat)  # [T, Q, 1, W]
        elif self.similarity_method == 'sqr':
            pass

        logits = logits.squeeze(-2).view(-1, self.way_num)

        return logits
