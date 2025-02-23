import torch
import torchvision.models as models
import torch.nn as nn

def get_model(model_name="mobilenet", num_classes=2):
    """
    Loads a pre-trained model (MobileNetV2 or Xception) and modifies it for binary classification.

    :param model_name: "mobilenet" or "xception"
    :param num_classes: Number of output classes (default: 2 - real/fake)
    :return: PyTorch model
    """
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "xception":
        model = models.xception(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError("Invalid model name. Choose 'mobilenet' or 'xception'.")

    return model

if __name__ == "__main__":
    # Test the model loading
    model_mobilenet = get_model("mobilenet")
    model_xception = get_model("xception")
    print("Models loaded successfully!")
