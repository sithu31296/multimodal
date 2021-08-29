import torch
import os
from PIL import Image
from torchvision.datasets import CIFAR10
from multimodal import clip


@torch.no_grad()
def simple_example():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = clip.load('ViT-B/32', jit=True)

    image = Image.open('Clip.png').convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)

    text = ['a diagram', 'a dog', 'a car']
    text = clip.tokenize(text).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_logits, text_logits = model(image, text)
    probs = image_logits.softmax(dim=-1).cpu().numpy()

    print("Label probs: ", probs)


@torch.no_grad()
def zero_shot_prediction():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = clip.load('ViT-B/32', jit=True)

    cifar10 = CIFAR10(os.path.expanduser("~/.cache"), download=True, train=False)

    # prepare the inputs
    image, class_id = cifar10[10]
    image = preprocess(image).unsqueeze(0).to(device)
    text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10.classes]).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # print the result
    print("Top predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar10.classes[index]:16s}: {100* value.item():.2f}%")


