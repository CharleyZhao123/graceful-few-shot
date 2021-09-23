import torch

# 划分数据为meta-task形式
def split_shot_query(data, way, shot, query, task_per_batch=1):
    img_shape = data.shape[1:]
    data = data.view(task_per_batch, way, shot + query, *img_shape)
    x_shot, x_query = data.split([shot, query], dim=2)
    x_shot = x_shot.contiguous()
    x_query = x_query.contiguous().view(task_per_batch, way * query, *img_shape)
    return x_shot, x_query

# 得到meta-task形式的label
def make_nk_label(n, k, task_per_batch=1):
    label = torch.arange(n).unsqueeze(1).expand(n, k).reshape(-1)
    label = label.repeat(task_per_batch)
    return label

