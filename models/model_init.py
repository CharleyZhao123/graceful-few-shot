# from utils import check_args
import torch
import clip
import sys
sys.path.append('..')

model_list = {}

def model_register(name):
    '''
    模型注册器
    '''
    def decorator(cls):
        model_list[name] = cls
        return cls
    return decorator

def build_model(model_name, model_args=None, model_load_para=None, **kwargs):
    '''
    构建神经网络模型
    输入: 神经网络模型各参数
    输出: 初始化好参数(根据存储的参数或者随机初始化参数)的神经网络模型
    '''

    # ===== 构建模型 =====
    # 直接从整体参数中构建, 不建议
    if model_load_para.get('load'):
        model_para = torch.load(model_load_para['load'])
        model = model_list[model_para['model']](**model_para['model_args'])
        model.load_state_dict(model_para['model_sd'])
    elif model_name == 'clip':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load('ViT-B/32', device)
        return model
    # 构建随机初始化模型
    else:
        model = model_list[model_name](**model_args)

    # ===== 选择加载子模块参数 =====
    if model_load_para.get('load_encoder'):
        encoder_para = torch.load(model_load_para['load_encoder'])
        encoder = model_list[encoder_para['model']](**encoder_para['model_args']).encoder
        model.encoder.load_state_dict(encoder.state_dict())

    if model_load_para.get('load_classifier'):
        classifier_para = torch.load(model_load_para['load_classifier'])
        classifier = model_list[classifier_para['model']](**classifier_para['model_args']).classifier
        model.classifier.load_state_dict(classifier.state_dict())

    if model_load_para.get('load_pure_encoder'):
        encoder_para = torch.load(model_load_para['load_pure_encoder'])
        model.encoder.load_state_dict(encoder_para)
    
    # 加载旧工程中的参数
    if model_load_para.get('load_old_encoder'):
        model_para = torch.load(model_load_para['load_old_encoder'])['model_sd']
        encoder_dict = {}
        for k, v in model_para.items():
            if k[:8] == "encoder.":
                k = k[8:]
                encoder_dict[k] = v
        
        if 'encoder' in dir(model):
            model.encoder.load_state_dict(encoder_dict)
        else:
            model.load_state_dict(encoder_dict)
    
    # ===== 其他 =====
    if torch.cuda.is_available():
        model.cuda()

    return model


if __name__ == '__main__':
    network_args = {
        'model_name': 'resnet12',
        'model_args': {},
        'model_load_para':
        {
            'load_encoder': '/space1/zhaoqing/code/graceful-few-shot/models/backbone/pretrained/resnet18-f37072fd.pth',
        },
        'similarity_method': 'cos'  # 'cos', 'sqr'
    }
    model_para = torch.load('/space1/zhaoqing/code/few_shot_meta_baseline/save/pre_meta_2_stage/linear/metabasepre2/max-tva.pth')['model_sd']
