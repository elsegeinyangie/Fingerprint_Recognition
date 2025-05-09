from models.resnet import resnet
from models.vgg import vgg
from models.mobilenet import mobilenetmodel


def get_model(model_name, input_shape):
    """Get the specified model (ResNet, VGG, MobileNet)"""
    if model_name == 'resnet':
        return resnet(input_shape)
    elif model_name == 'vgg':
        return vgg(input_shape)
    elif model_name == 'mobilenet':
        return mobilenetmodel(input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={
            "gender_output": "categorical_crossentropy",
            "hand_output": "categorical_crossentropy",
            "finger_output": "categorical_crossentropy"
        },
        metrics={
            "gender_output": "accuracy",
            "hand_output": "accuracy",
            "finger_output": "accuracy"
        }
    )
    return model
