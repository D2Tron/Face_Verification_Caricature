#   Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms
import math
import random
import wandb

#Random seed
random.seed(3)

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

image = torchvision.io.read_image("./black.jpg", mode=torchvision.io.ImageReadMode.RGB)
transform = transforms.Resize((218, 178), antialias=True)
image = transform(image)

#Instatitate the model, loss function, and the optimizer
model = AlteredNet(18).to(device)
model.load_state_dict(torch.load("./ver_model_coop.pt"))
#model.load_state_dict(torch.load("./car_model.pt"))

fspace = []

model.remove_last_layer()

model.eval()
with torch.no_grad():
    data = image.to(device).float()
    data = torch.unsqueeze(data, 0)
    output = model(data)
    
    # output = model.sigmoid_layer(output)
    output = output.detach().cpu().numpy()

    while len(output.shape) > 2:
        output = output.squeeze(-1)

    fspace.append(output)

fspace = np.concatenate(fspace, axis=0)

np.save('fspace_Static.npy', fspace)