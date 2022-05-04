import os

path = './'
file_list = os.listdir(path)
print(file_list)

for i in file_list:
    if i[-2:] == '_w' or i[-2:] == '_g':
        sub_path = path + i + '/'
        sub_list = os.listdir(sub_path)
        n = 0
        for j in sub_list:
            old_name = sub_path + j
            new_name = sub_path + i + '_' + str(n) + '.jpg'
            n += 1
            os.rename(old_name, new_name)
        