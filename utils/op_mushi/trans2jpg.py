import os
from PIL import Image

def trans2jpg(path):
    image_list = os.listdir(path)
    for image in image_list:
        if 'D' in image:
            continue
        if '.png' in image:
            image_path = os.path.join(path, image)
            img = Image.open(image_path).convert('RGB')
            img.save(os.path.splitext(image_path)[0] + '.jpg')

if __name__ == '__main__':
    dataset_floader_path = '/Users/charleyzhao/code/dataset/mushi/mushi-t9/'
    class_name = 'car'
    path = os.path.join(dataset_floader_path, class_name)

    trans2jpg(path)
