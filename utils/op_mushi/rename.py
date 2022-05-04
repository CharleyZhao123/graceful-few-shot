import os

def rename(new_data_path, begin_id):
    file_list = os.listdir(new_data_path)
    for f in file_list:
        old_name = os.path.join(new_data_path, f)
        if 'json' in f:  # label_0.json
            old_id = int(f[6:-5])
            new_f = f.replace(f[6:-5], str(begin_id+old_id))
        elif 'mask' in f:  # mask_12_T-90_5.png
            if 'SUV' in f:
                old_id = int(f[5:-10])
                new_f = f.replace(f[5:-10], str(begin_id+old_id))
            elif 'T-90' in f:
                old_id = int(f[5:-11])
                new_f = f.replace(f[5:-11], str(begin_id+old_id))
        elif 'origin' in f:  # origin_43.png
            old_id = int(f[7:-4])
            new_f = f.replace(f[7:-4], str(begin_id+old_id))
        else: # label_0.png
            old_id = int(f[6:-4])
            new_f = f.replace(f[6:-4], str(begin_id+old_id))
        
        new_name = os.path.join(new_data_path, new_f)
        os.rename(old_name, new_name)
        print(old_name, ' =====> ', new_name)

    return None

def rename_2(path):
    image_list = os.listdir(path)
    id = 100
    for image in image_list:
        if 'D' in image:
            continue
        id = id + 1
        new_name = str(id) + '.jpg'
        old_path = os.path.join(path, image)
        new_path = os.path.join(path, new_name)
        
        os.rename(old_path, new_path)



if __name__ == '__main__':
    # new_data_path = '/Users/charleyzhao/code/dataset/animals_origin/T-90+SUV'
    # begin_id = 672

    # rename(new_data_path, begin_id)

    path = '/Users/charleyzhao/code/dataset/mushi/mushi-t6/tank'
    rename_2(path)

