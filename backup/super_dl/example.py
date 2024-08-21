import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Define the transforms for data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create ImageNet iterable dataset
imagenet_dataset = ImageFolder(root='classification/datasets/cifar-10/train', transform=transform)

# Create DataLoader for the iterable dataset
batch_size = 32
num_workers = 4
imagenet_loader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Iterate through the dataset during training
for batch_idx, (data, target) in enumerate(imagenet_loader):
    # Your training code here
    # 'data' contains the input images
    # 'target' contains the corresponding labels
    pass
