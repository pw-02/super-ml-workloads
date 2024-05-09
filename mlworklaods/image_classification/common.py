import torchvision
from lightning.fabric import Fabric


# Make a model given the name
def make_model(fabric: Fabric, model_name: str):
    if model_name in torchvision.models.list_models():
        with fabric.init_module(empty_init=True):
            return torchvision.models.get_model(model_name)
    raise Exception(f"Unknown model: {model_name}")
