from PIL import Image

with open('data/cifar10/train/Horse/walking_horse_s_002129.png', 'rb') as f:
    data = Image.open(f).convert("RGB")
    data.show()