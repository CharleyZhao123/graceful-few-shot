import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dog_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_100.jpg")).unsqueeze(0).to(device)
cow_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_55.jpg")).unsqueeze(0).to(device)
cat_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_5.jpg")).unsqueeze(0).to(device)
goat_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_40.jpg")).unsqueeze(0).to(device)
rabbit_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_19.jpg")).unsqueeze(0).to(device)
pig_image = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_g/cat_g_19.jpg")).unsqueeze(0).to(device)

pig_image1 = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_w/cat_w_20.jpg")).unsqueeze(0).to(device)
pig_image2 = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_w/cat_w_40.jpg")).unsqueeze(0).to(device)

pig_image3 = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_t/cat_t_0.png")).unsqueeze(0).to(device)
pig_image4 = preprocess(Image.open(
    "/space1/zhaoqing/dataset/fsl/animals/cat_s/cat_s_0.png")).unsqueeze(0).to(device)


text = clip.tokenize(["A photo of a cat"]).to(device)

with torch.no_grad():
    dog_image_features = model.encode_image(dog_image)
    cow_image_features = model.encode_image(cow_image)
    cat_image_features = model.encode_image(cat_image)
    goat_image_features = model.encode_image(goat_image)
    rabbit_image_features = model.encode_image(rabbit_image)
    pig_image_features = model.encode_image(pig_image)
    pig_image1_features = model.encode_image(pig_image1)
    pig_image2_features = model.encode_image(pig_image2)
    pig_image3_features = model.encode_image(pig_image3)
    pig_image4_features = model.encode_image(pig_image4)

    # [5, 512]
    image_features = torch.cat((dog_image_features, cow_image_features, cat_image_features, goat_image_features, rabbit_image_features,
                                pig_image_features, pig_image1_features, pig_image2_features, pig_image3_features, pig_image4_features), dim=0)

    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print(similarity)
    # values, indices = similarity[0]

    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
