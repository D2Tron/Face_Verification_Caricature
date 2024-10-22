import numpy as np
import torch
from torch import nn
import torchvision
    
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

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

#Preprocessing


def prom_feature_analyzer(image, group, opinion):
    model = AlteredNet(18)
    model.load_state_dict(torch.load("./ver_model_coop.pt"))
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        data = image.to(device).float()
        data = torch.unsqueeze(data, 0)
        output = model(data)

        output = model.sigmoid_layer(output)
        output = output.detach().cpu().numpy().flatten()

        if group == "conforming":
            for i in range(len(output)):
                if output[i] >= .35 and output[i] <= .65:
                    output[i] = opinion[i]
        else:
            for i in range(len(output)):
                if output[i] >= .35 and output[i] <= .65:
                    output[i] = 1 - opinion[i]

        output = np.round(output)

        return output

image = torchvision.io.read_image("./ourcar/robert_downey_jr./02.jpg", mode=torchvision.io.ImageReadMode.RGB)
opinion = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0])
group = "conforming"

result = prom_feature_analyzer(image, group, opinion)

features = np.array(['cheekbones', 'cheeks', 'chin', 'ears', 'eyebrows', 'eyelids', 'eyes', 'facial hair', 'forehead',
                     'hair', 'head', 'lips', 'mouth', 'neck', 'nose', 'skin', 'teeth', 'upper lip'])

features_opinion_p = [f for f, binary in zip(features, opinion) if binary == 1]
print(f"Your prominent features according to you are: {', '.join(features_opinion_p)}\n")

features_opinion_o = [f for f, binary in zip(features, opinion) if binary == 0]
print(f"Your non-prominent features according to you are: {', '.join(features_opinion_o)}\n")

features_prominent = [f for f, binary in zip(features, result) if binary == 1]
print(f"Your prominent features according to the model are: {', '.join(features_prominent)}\n")

features_obscure = [f for f, binary in zip(features, result) if binary == 0]
print(f"Your non-prominent features according to the model are: {', '.join(features_obscure)}")