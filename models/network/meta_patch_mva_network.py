# changed from https://github.com/miguealanmath/MAML-Pytorch/blob/master/MAML-v1.ipynb
from numpy import outer
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models import model_register, build_model
import utils
import utils.few_shot as fs

@model_register('meta-patch-mva-network')
class MetaPatchMVANetwork(nn.Module):
    def __init__(self, encoder_name='resnet12', encoder_args={}, encoder_load_para={},
                 mva_name='dot-attention', mva_args={}, mva_load_para={}, task_info={},
                 similarity_method='cos', patch_mode='default', meta_info={}):
        super().__init__()
        # ===== 子模块构建 =====
        self.encoder = build_model(
            encoder_name, encoder_args, encoder_load_para)
        self.mva = build_model(mva_name, mva_args, mva_load_para)

        # ===== 任务信息获得 =====
        self.batch_size = task_info['batch_size']
        self.shot_num = task_info['shot_num']
        self.way_num = task_info['way_num']
        self.query_num = task_info['query_num']

        # ===== Patch模式信息 =====
        # patch_mode:
        # default: 当做对Support的数据增强看待
        self.patch_mode = patch_mode

        # patch_num: Patch的数量, 为图像Patch维度, 第0个图像为原图
        self.patch_num = 1

        # ===== 元学习信息 =====
        self.meta_info = meta_info

        # ===== 其他 =====
        self.similarity_method = similarity_method
        self.mva_name = mva_name
        self.mva_args = mva_args

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

    def feature_augmentation(self, fkey, w_index, s_index, aug_type='none'):
        '''
        对特征进行增强
        '''
        if aug_type == 'none':
            query_feat = fkey[0, w_index, s_index, :].clone().detach()
            return query_feat
        elif aug_type == 'zero':
            query_feat = fkey[0, w_index, s_index, :].clone().detach()
            zero_feat = torch.zeros_like(query_feat)
            return zero_feat
        elif aug_type == 'random':
            query_feat = fkey[0, w_index, s_index, :].clone().detach()
            random_mask = torch.randn_like(query_feat).cuda()
            random_feat = query_feat + random_mask * 0.07
            return random_feat
        elif aug_type == 'way_mean':
            way_feat = fkey[0, w_index, :, :].clone().detach()  # [W, dim]
            way_mean_feat = torch.mean(way_feat, dim=0)
            return way_mean_feat
        elif aug_type == 'way_other_mean':
            shot_num = fkey.shape[2]

            way_feat = fkey[0, w_index, :, :].clone().detach()  # [W, dim]

            index = [i for i in range(shot_num)]
            index.remove(s_index)
            index = torch.tensor(index, dtype=torch.long).cuda()

            way_other_feat = torch.index_select(way_feat, 0, index)
            way_other_mean_feat = way_other_feat.mean(dim=0)
            return way_other_mean_feat

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
                aug_feat = self.feature_augmentation(
                    fkey, w_index, s_index, aug_type)
                fkey[0, w_index, s_index, :] = aug_feat

        fquery = torch.stack(fquery, dim=0).unsqueeze(0).cuda()
        flabel = torch.tensor(flabel, dtype=torch.long).cuda()

        return fkey, fquery, flabel

    def meta_train(self, key, query, meta_info={}, enhance_threshold=0.0, enhance_top=10):
        '''
        使用support set数据(key)进行inner-loop训练
        epoch_num: 正常训练的epoch次数, 每个epoch构建的tasks是不同的, 有难有易
        enhance_threshold: 判断是否需要对该epoch增强的阈值, 如果acc小于该阈值, 则增强
        enhance_top: 增强次数上限, 最多增强该次数(包含)
        '''
        # 关键超参
        inner_epoch_num = meta_info['inner_epoch']
        lr = meta_info['inner_lr']
        aug_type = meta_info['inner_aug_type']
        choice_num = meta_info['inner_choice_num']
        l1 = meta_info['inner_use_l1']

        outer_epoch_num = meta_info['outer_epoch']
        outer_optimizer_name = meta_info['outer_optimizer_name']
        outer_optimizer_args = meta_info['outer_optimizer_args']

        # 1: 第一次inner更新
        epoch = 1
        enhance_num = 0
        fkey, fquery, flabel = self.build_fake_trainset(
            key, choice_num=choice_num, aug_type=aug_type, epoch=epoch)
        proto_feat = self.mva(fquery, fkey)
        logits = self.get_logits(proto_feat, fquery)
        loss = F.cross_entropy(logits, flabel)
        acc = utils.compute_acc(logits, flabel)

        grad = torch.autograd.grad(loss, self.mva.parameters())
        tuples = zip(grad, self.mva.parameters())
        fast_weights = list(map(lambda p: p[1] - lr * p[0], tuples))
        epoch += 1

        # 2: 第[2: inner_epoch]次更新
        while epoch < inner_epoch_num + 1:
            fkey, fquery, flabel = self.build_fake_trainset(
                key, choice_num=choice_num, aug_type=aug_type, epoch=epoch)
            proto_feat = self.mva(fquery, fkey, fast_weights)
            logits = self.get_logits(proto_feat, fquery)
            loss = F.cross_entropy(logits, flabel)
            acc = utils.compute_acc(logits, flabel)

            # print('mva train epoch: {} acc={:.2f} loss={:.2f}'.format(
            #     epoch, acc, loss))

            grad = torch.autograd.grad(loss, self.mva.parameters())
            tuples = zip(grad, self.mva.parameters())
            fast_weights = list(map(lambda p: p[1] - lr * p[0], tuples))

            if acc < enhance_threshold:
                enhance_num += 1
                if enhance_num > enhance_top:
                    enhance_num = 0
                    epoch += 1
                else:
                    pass
                    # print('mva train epoch enhance time: {}'.format(
                    #     enhance_num))
            else:
                enhance_num = 0
                epoch += 1
        
        # 3: 对query进行评估, 并outer更新模型
        proto_feat = self.mva(query, key, fast_weights)

        # print("meta inner loop train done.")

    def forward(self, image):
        # ===== 数据整理 =====
        # Patch image shape: [320, 10, 3, 80, 80]
        self.patch_num = image.shape[1]
        image_num = image.shape[0]

        # [320x10, 3, 80, 80]
        image = image.reshape(-1,
                              image.shape[2], image.shape[3], image.shape[4])

        # ===== 提取特征 =====
        with torch.no_grad():
            if 'encode_image' in dir(self.encoder):
                feature = self.encoder.encode_image(image).float()
            else:
                feature = self.encoder(image)

        # ===== 整理, 划分特征 =====
        feat_dim = feature.shape[-1]
        feature = feature.reshape(
            image_num, self.patch_num, feat_dim)  # [320, 10, 512]

        # shot_feat: [T, W, S, P, dim], query_feat: [T, Q, P, dim], P: patch_num
        shot_feat, query_feat = fs.split_shot_query(
            feature, self.way_num, self.shot_num, self.query_num, self.batch_size)

        # 根据patch_mode整理特征
        # 'default': qurey取原图特征, support将Patch看作数据增广
        if self.patch_mode == 'default':
            new_shot_num = self.shot_num * self.patch_num
            # 相当于将S shot变换为SxP shot: [T, W, S, P, dim] -> [T, W, SxP, dim]
            shot_feat = shot_feat.reshape(
                self.batch_size, self.way_num, new_shot_num, feat_dim)
            # [T, Q, P, dim] -> [T, Q, dim]
            query_feat = query_feat[:, :, 0, :]
        else:
            pass

        # ===== 元训练 =====
        # 元训练模式的情况下, 每个batch的任务数只能为1
        self.meta_train(key=shot_feat, query=query_feat, meta_info=self.meta_info,
                              enhance_threshold=0.0, enhance_top=20)

        # ===== 将特征送入MVA进行计算 =====
        # proto_feat: [T, Q, W, dim]
        proto_feat = self.mva(query_feat, shot_feat)

        # ===== 得到最终的分类logits =====
        logits = self.get_logits(proto_feat, query_feat)

        return logits
