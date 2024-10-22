#   Imports
import torch
from torch import nn
import torchvision
from scipy.spatial.distance import cosine
import numpy as np

#Device declaration
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

#Model class implementation
class AlteredNet(nn.Module):
    def __init__(self, num_out_features):
        super(AlteredNet, self).__init__()

        # self.softmax_layer = nn.Softmax(dim=1)
        self.sigmoid_layer = nn.Sigmoid()

        self.model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_out_features)

    def remove_last_layer(self):
        # Remove the last layer by modifying self.model
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        output = self.model(x)
        return output

class AlteredNet2(nn.Module):
    def __init__(self, num_out_features):
        super(AlteredNet2, self).__init__()

        # self.softmax_layer = nn.Softmax(dim=1)
        self.sigmoid_layer = nn.Sigmoid()

        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_out_features)

    def remove_last_layer(self):
        # Remove the last layer by modifying self.model
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        output = self.model(x)
        return output

def compare_models(model1, model2):
    model1_params = dict(model1.named_parameters())
    model2_params = dict(model2.named_parameters())

    for name, params1 in model1_params.items():
        if name in model2_params:
            params2 = model2_params[name]

            similarity = 1 - cosine(params1.data.cpu().numpy().flatten(), params2.data.cpu().numpy().flatten())

            print(f"Cosine Similarity for layer {name}: {similarity}")


# Example usage:
model_ver = AlteredNet(18).to(device)
model_car = AlteredNet(18).to(device)

#model_ver.load_state_dict(torch.load("./ver_sub_model.pt"))
model_car.load_state_dict(torch.load("./car_model.pt"))

compare_models(model_ver, model_car)