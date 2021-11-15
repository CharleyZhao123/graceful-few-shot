import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import model_register, build_model
import utils
import utils.few_shot as fs


@model_register('mva-network')
class MVANetwork(nn.Module):
    def __init__(self, encoder_name='resnet12', encoder_args={}, encoder_load_para={},
                 mva_name='dot-attention', mva_args={'update': False}, mva_load_para={}, task_info={},
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
        self.mva_name = mva_name
        self.mva_update = mva_args['update']

    def get_logits(self, proto_feat, query_feat):
        '''
        得到最终的分类logits
        '''
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

    def feature_augmentation(self, feat, aug_type='none'):
        '''
        对特征进行增强
        '''
        if aug_type == 'none':
            return feat
        elif aug_type == 'zero':
            zero_feat = torch.zeros_like(feat)
            return zero_feat

    def build_fake_trainset(self, key, choice_type='random', choice_num=3, aug_type='none', epoch=0):
        '''
        利用support set数据(key)构建假训练集
        输入: key: [1, W, S, dim]
        输出:
            fkey: [1, W, S, dim]
            fquery: [1, Q, dim]
            flabel: [Q, W]
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
                w_choice_id_list = random.sample(
                    range(0, shot_num), choice_num)
                choice_id_list.append(w_choice_id_list)  # [[...], [], ...]

        # 根据choice_id_list构建假训练集
        for w_index, w_choice_id_list in enumerate(choice_id_list):
            for s_index in w_choice_id_list:
                # print(w_index, ' ', s_index)
                # 得到用作query的key特征
                query_feat = fkey[0, w_index, s_index, :].clone().detach()
                fquery.append(query_feat)

                # 构建label
                query_label = w_index
                flabel.append(query_label)

                # 增强对应的原key特征
                aug_feat = self.feature_augmentation(query_feat, aug_type)
                fkey[0, w_index, s_index, :] = aug_feat

        fquery = torch.stack(fquery, dim=0).unsqueeze(0).cuda()
        flabel = torch.tensor(flabel, dtype=torch.long).cuda()

        return fkey, fquery, flabel

    def train_mva(self, key, epoch_num=30):
        '''
        使用support set数据(key)训练mva
        '''
        optimizer = torch.optim.SGD(self.mva.parameters(), lr=1e-3,
                                    momentum=0.9, dampening=0.9, weight_decay=0)
        with torch.enable_grad():
            for epoch in range(1, epoch_num+1):
                fkey, fquery, flabel = self.build_fake_trainset(
                    key, choice_num=1, aug_type='zero', epoch=epoch)

                optimizer.zero_grad()

                proto_feat = self.mva(fquery, fkey)
                logits = self.get_logits(proto_feat, fquery)
                loss = F.cross_entropy(logits, flabel)
                acc = utils.compute_acc(logits, flabel)

                loss.backward()
                optimizer.step()

                print('mva train epoch: {} acc={:.2f} loss={:.2f}'.format(epoch, acc, loss))
        
        print("mva train done.")


    def forward(self, image):
        # ===== 提取特征并整理 =====
        if 'encode_image' in dir(self.encoder):
            feature = self.encoder.encode_image(image)
        else:
            feature = self.encoder(image)

        # 划分特征
        # shot_feat: [T, W, S, dim]
        # query_feat: [T, Q, dim]
        shot_feat, query_feat = fs.split_shot_query(
            feature, self.way_num, self.shot_num, self.query_num, self.batch_size)

        # ===== MVA训练 =====
        if self.mva_update:
            self.train_mva(key=shot_feat, epoch_num=10)
        
        # ===== 将特征送入MVA进行计算 =====
        # proto_feat: [T, Q, W, dim]
        proto_feat = self.mva(query_feat, shot_feat)

        # ===== 得到最终的分类logits =====
        logits = self.get_logits(proto_feat, query_feat)

        return logits
