import json
from PIL import Image
from torchvision import transforms

def serialize_torchvision_transformation(transform):
    # Convert the transform to a list of dictionaries
    serialized_transform = []

    if isinstance(transform, transforms.Compose):
        for t in transform.transforms:
            serialized_transform.append(serialize_torchvision_transformation(t))
    else:
        transform_class_name = transform.__class__.__name__
        transform_args = transform.__dict__

        # Convert sets to lists, handle nested dictionaries
        for key, value in transform_args.items():
            if isinstance(value, set):
                transform_args[key] = list(value)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, set):
                        value[nested_key] = list(nested_value)

            # Convert InterpolationMode to string
            if isinstance(value, transforms.InterpolationMode):
                transform_args[key] = value.value

        serialized_transform.append({
            'type': transform_class_name,
            'args': transform_args
        })

    return serialized_transform

def deserialize_torchvision_transformation(serialized_transform):
    # Reconstruct the transform object from the serialized representation
    transform_list = []

    for t in serialized_transform:
        transform_class_name = t[0].get('type')
        transform_args = t[0].get('args', {})

        # Convert InterpolationMode string back to enum
        for key, value in transform_args.items():
            if key == 'interpolation' and isinstance(value, str):
                transform_args[key] = getattr(transforms.InterpolationMode, value)

        try:
            transform_class = getattr(transforms, transform_class_name)
            transform_instance = transform_class(**transform_args)
            transform_list.append(transform_instance)
        except AttributeError:
            print(f"Warning: Transform type '{transform_class_name}' not found.")

    return transforms.Compose(transform_list)

# Your PyTorch transform
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Serialize the transform
serialized_transform = serialize_torchvision_transformation(data_transform)

# Convert the serialized transform to a JSON string
transform_json = json.dumps(serialized_transform)

# Convert the JSON string to a list of dictionaries
serialized_transform = json.loads(transform_json)

# Recreate the PyTorch transform object from the list of dictionaries
recreated_transform = deserialize_torchvision_transformation(serialized_transform)

# Sample usage with an image (uncomment these lines as needed)
# img = Image.open('datasets/vision/cifar-10/train/Airplane/aeroplane_s_000004.png')
# img_transformed = recreated_transform(img)
# img_transformed.show()