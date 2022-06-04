import utils
utils.set_gpu('7')
import torch
import clip
from PIL import Image
import os
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
base_path = "/space1/zhaoqing/dataset/fsl/img-mini-imagenet/images"

image1 = preprocess(Image.open(os.path.join(base_path, "n0153282900000469.jpg"))).unsqueeze(0).to(device)
image2 = preprocess(Image.open(os.path.join(base_path, "n0153282900000474.jpg"))).unsqueeze(0).to(device)
image3 = preprocess(Image.open(os.path.join(base_path, "n0153282900000525.jpg"))).unsqueeze(0).to(device)
image4 = preprocess(Image.open(os.path.join(base_path, "n0461250400001154.jpg"))).unsqueeze(0).to(device)
image5 = preprocess(Image.open(os.path.join(base_path, "n0679411000000127.jpg"))).unsqueeze(0).to(device)
image6 = preprocess(Image.open(os.path.join(base_path, "n0377554600000587.jpg"))).unsqueeze(0).to(device)
image7 = preprocess(Image.open(os.path.join(base_path, "n0377554600000619.jpg"))).unsqueeze(0).to(device)
image8 = preprocess(Image.open(os.path.join(base_path, "n1313361300001270.jpg"))).unsqueeze(0).to(device)
image9 = preprocess(Image.open(os.path.join(base_path, "n1313361300001299.jpg"))).unsqueeze(0).to(device)
image10 = preprocess(Image.open(os.path.join(base_path, "n1313361300001198.jpg"))).unsqueeze(0).to(device)
image11 = preprocess(Image.open(os.path.join(base_path, "n0306224500000604.jpg"))).unsqueeze(0).to(device)
image12 = preprocess(Image.open(os.path.join(base_path, "n0306224500000606.jpg"))).unsqueeze(0).to(device)
image13 = preprocess(Image.open(os.path.join(base_path, "n0207436700000281.jpg"))).unsqueeze(0).to(device)
image14 = preprocess(Image.open(os.path.join(base_path, "n0185567200000826.jpg"))).unsqueeze(0).to(device)
image15 = preprocess(Image.open(os.path.join(base_path, "n0191074700000599.jpg"))).unsqueeze(0).to(device)
image16 = preprocess(Image.open(os.path.join(base_path, "n0209124400000036.jpg"))).unsqueeze(0).to(device)
image17 = preprocess(Image.open(os.path.join(base_path, "n0341704200000718.jpg"))).unsqueeze(0).to(device)
image18 = preprocess(Image.open(os.path.join(base_path, "n0347668400000190.jpg"))).unsqueeze(0).to(device)
image19 = preprocess(Image.open(os.path.join(base_path, "n0353578000000468.jpg"))).unsqueeze(0).to(device)
image20 = preprocess(Image.open(os.path.join(base_path, "n0353578000000473.jpg"))).unsqueeze(0).to(device)
image21 = preprocess(Image.open(os.path.join(base_path, "n0377350400000159.jpg"))).unsqueeze(0).to(device)
image22 = preprocess(Image.open(os.path.join(base_path, "n0377350400000171.jpg"))).unsqueeze(0).to(device)
image23 = preprocess(Image.open(os.path.join(base_path, "n0451500300000645.jpg"))).unsqueeze(0).to(device)
image24 = preprocess(Image.open(os.path.join(base_path, "n0451500300000662.jpg"))).unsqueeze(0).to(device)
image25 = preprocess(Image.open(os.path.join(base_path, "n0459674200000409.jpg"))).unsqueeze(0).to(device)
image26 = preprocess(Image.open(os.path.join(base_path, "n0459674200000387.jpg"))).unsqueeze(0).to(device)

text = clip.tokenize(["A photo of metal"]).to(device)

with torch.no_grad():
    image1_feature = model.encode_image(image1)
    image2_feature = model.encode_image(image2)
    image3_feature = model.encode_image(image3)
    image4_feature = model.encode_image(image4)
    image5_feature = model.encode_image(image5)
    image6_feature = model.encode_image(image6)
    image7_feature = model.encode_image(image7)
    image8_feature = model.encode_image(image8)
    image9_feature = model.encode_image(image9)
    image10_feature = model.encode_image(image10)
    image11_feature = model.encode_image(image11)
    image12_feature = model.encode_image(image12)
    image13_feature = model.encode_image(image13)
    image14_feature = model.encode_image(image14)
    image15_feature = model.encode_image(image15)
    image16_feature = model.encode_image(image16)
    image17_feature = model.encode_image(image17)
    image18_feature = model.encode_image(image18)
    image19_feature = model.encode_image(image19)
    image20_feature = model.encode_image(image20)
    image21_feature = model.encode_image(image21)
    image22_feature = model.encode_image(image22)
    image23_feature = model.encode_image(image23)
    image24_feature = model.encode_image(image24)
    image25_feature = model.encode_image(image25)
    image26_feature = model.encode_image(image26)

    # [5, 512]
    image_features = torch.cat((image1_feature, image2_feature, image3_feature, image4_feature, image5_feature, 
                                image6_feature, image7_feature, image8_feature, image9_feature, image10_feature,
                                image11_feature, image12_feature, image13_feature, image14_feature, image15_feature,
                                image16_feature, image17_feature, image18_feature, image19_feature, image20_feature,
                                image21_feature, image22_feature, image23_feature, image24_feature, image25_feature, image26_feature), dim=0)

    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T)  # .softmax(dim=-1)

    similarity = similarity.squeeze().tolist()

    print(similarity)

    dataframe = pd.DataFrame({'sim': similarity})

    dataframe.to_csv('output_sim.csv', index=False, sep=',')

    # values, indices = similarity[0]

    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
