import torchvision.transforms as transforms
import json
from PIL import Image
import torch
def transform_to_dict(transform):
    transform_dict = {}

    for tf in transform.transforms:
        transform_name = tf.__class__.__name__

        if transform_name == 'Resize':
            transform_dict[transform_name] = tf.size
        elif transform_name == 'Normalize':
            transform_dict[transform_name] = {'mean': tf.mean, 'std': tf.std}
        else:
            transform_dict[transform_name] = None

    return transform_dict

def dict_to_transform(transform_dict):
    transform_list = []

    for transform_name, params in transform_dict.items():
        if transform_name == 'Resize':
            transform_list.append(transforms.Resize(params))
        elif transform_name == 'Normalize':
            transform_list.append(transforms.Normalize(mean=params['mean'], std=params['std']))
        elif params is None:
            transform_list.append(getattr(transforms, transform_name)())
        else:
            raise ValueError(f"Unsupported transform: {transform_name}")

    return transforms.Compose(transform_list)


# Your PyTorch transform
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Convert PyTorch transform to dictionary
transform_dict = transform_to_dict(data_transform)
# Convert the transform_dict to JSON
transform_json = json.dumps(transform_dict, indent=2)


# Load the JSON back to dictionary
loaded_transform_dict = json.loads(transform_json)
# Recreate PyTorch transform from dictionary
recreated_transform = dict_to_transform(transform_dict)

print(recreated_transform)
# Sample usage with an image (uncomment these lines as needed)
img = Image.open('datasets/vision/cifar-10/train/Airplane/aeroplane_s_000004.png')
img_transformed1 = data_transform(img)
img_transformed2 = recreated_transform(img)

is_equal_12 = torch.equal(img_transformed1, img_transformed2)
