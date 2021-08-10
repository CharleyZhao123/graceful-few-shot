import torch
import numpy as np

a = [1, 4, 6, 5, 2, 3, 7]
b = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

tensor_a = torch.tensor(a)
sorted_a, index = torch.sort(tensor_a)
print(tensor_a)
print(sorted_a)
print(index)

list_index = index.numpy().tolist()
print(list_index)

sorted_b = b[list_index]
