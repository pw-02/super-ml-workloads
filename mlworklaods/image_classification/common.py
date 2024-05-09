import torchvision
from lightning.fabric import Fabric
import torchvision.transforms as transforms
from torch import  Tensor, no_grad

# Make a model given the name
def make_model(fabric: Fabric, model_name: str):
    if model_name in torchvision.models.list_models():
        with fabric.init_module(empty_init=True):
            return torchvision.models.get_model(model_name)
    raise Exception(f"Unknown model: {model_name}")


# Transformation function for data augmentation
def transform():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

# Calculate accuracy for top-k predictions
def accuracy(output: Tensor, target: Tensor, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k."""
    with no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
   
