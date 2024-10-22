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
random.seed(42)

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

#Dataloader class implementation
class CaricatureDataset(Dataset):
    def __init__(self, labels_file, root_dir, split, transform=None):
        self.annotations = pd.read_csv(labels_file, index_col=0)
        self.root_dir = root_dir
        self.split = split

        self.image_paths = []
        self.identities = self.annotations.axes[0].tolist()

        for id in self.identities:
            # Testing on
            # folder_name = id + "_caricature"
            folder_name = id

            image_names = os.listdir(os.path.join(root_dir, folder_name))
            random.shuffle(image_names)
            for img_index, img in enumerate(image_names):
                if split=="Train" and img_index<4:
                    self.image_paths.append((os.path.join(root_dir, folder_name, img)))
                elif split=="Test" and img_index==4:
                    self.image_paths.append((os.path.join(root_dir, folder_name, img)))

        self.transform = transform
    
    def __len__(self):
        # print(len(self.image_paths))
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if self.split=="Train":
            class_index = math.floor(idx/4)
        elif self.split=="Test":
            class_index = idx
        id = self.identities[class_index]
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        
        label = self.annotations.loc[id]

        return image, torch.tensor(label)

#Transformer
transform = transforms.Compose([
    transforms.Resize((218, 178), antialias=True)
])

#Create the datasets
train_dataset = CaricatureDataset(labels_file='binary_prom_labels.csv', root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Train", transform=transform)
test_dataset = CaricatureDataset(labels_file='binary_prom_labels.csv', root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Test", transform=transform)

#   Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Instatitate the model, loss function, and the optimizer
model = AlteredNet(18).to(device)
#Trained on
# model.load_state_dict(torch.load("./car_sub_model.pt"))
model.load_state_dict(torch.load("./car_model.pt"))

fspace = []

#model.remove_last_layer()

model.eval()
with torch.no_grad():
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device).float(), target.to(device).float()
        output = model(data)
        
        # output = model.sigmoid_layer(output)
        output = output.detach().cpu().numpy()

        while len(output.shape) > 2:
            output = output.squeeze(-1)

        fspace.append(output)

fspace = np.concatenate(fspace, axis=0)

np.save('outputs_ver_prom.npy', fspace)