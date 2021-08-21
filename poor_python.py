import pickle
import torch
import os

# imgname = ['sss']
# label_tensor = torch.zeros([1])
# data_tensor = torch.zeros([1, 3, 80, 80]).cuda()
# feature_tensor = torch.zeros([1, 512]).cuda()
# logits_tensor = torch.zeros([1, 64]).cuda()


# data_tensor = data_tensor.cpu()
# feature_tensor = feature_tensor.cpu()
# logits_tensor = logits_tensor.cpu()
# a = {
#     'imagename': imgname,
#     'label_tensor': label_tensor,
#     'data_tensor': data_tensor,
#     'feature_tensor': feature_tensor,
#     'logits_tensor': logits_tensor
# }

# gb_dataset_file = open('save/gb_dataset/a.pickle', 'wb')
# pickle.dump(a, gb_dataset_file)
# gb_dataset_file.close()
pickle_path = '/space1/zhaoqing/dataset/fsl/mini-imagenet/miniImageNet_category_split_train_phase_train.pickle'

with open(pickle_path, 'rb') as f:
    pack = pickle.load(f, encoding='latin1')

for k, v in pack.items():
    print(k)