# models/model_utils.py
from models.alexnet import alexnet
from models.googlenet import googlenet
from models.lenet import lenet
from models.resnet import resnet
from models.vgg import vgg
from models.mobilenet import mobilenetmodel


def get_model(model_name, input_shape):
    print(f"Loading model: {model_name} with input shape: {input_shape}")
    """Get the specified model"""
    if model_name == "alexnet":
        return alexnet(input_shape)
    elif model_name == "googlenet":
        return googlenet(input_shape)
    elif model_name == "lenet":
        return lenet(input_shape)
    elif model_name == "resnet":
        return resnet(input_shape)
    elif model_name == "vgg":
        return vgg(input_shape)
    elif model_name == "mobilenet":
        return mobilenetmodel(input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")