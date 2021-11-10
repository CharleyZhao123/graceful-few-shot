import pickle
import torch
import os

key = torch.rand(4, 15, 5, 512)
print(key[0, 0, 0, 0:10])
eye_base = torch.eye(512)
eye_repeat = eye_base.unsqueeze(0).repeat(15, 1, 1)
weight = eye_repeat

new_key = torch.tensordot(key, weight, dims=([3], [1]))
new_key = torch.matmul(key, weight)
print(new_key.shape)
print(new_key[0, 0, 0, 0:10])
