#   Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.metrics import f1_score
import math
import random

#Device declaration
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

#Define type of labels (prom/sub) and the type of dataset (caricature/veridical/combined)
label_type = "sub"
dataset_type = "caricature"

#Based on the type of dataset (car/ver), update the folder suffix
folder_suffix = ""
if dataset_type == "caricature":
    folder_suffix = "_caricature"

#Model class implementation
class AlteredNet(nn.Module):
    def __init__(self, num_out_features):
        super(AlteredNet, self).__init__()

        # self.softmax_layer = nn.Softmax(dim=1)
        self.sigmoid_layer = nn.Sigmoid()

        self.model = torchvision.models.resnet18()
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_out_features)

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
            folder_name = id + folder_suffix

            image_names = os.listdir(os.path.join(root_dir, folder_name))
            #Random seed
            random.seed(42)
            random.shuffle(image_names)
            for img_index, img in enumerate(image_names):
                if split=="Train" and img_index<3:
                    self.image_paths.append((os.path.join(root_dir, folder_name, img)))
                elif split=="Val" and img_index==3:
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
            class_index = math.floor(idx/3)
        elif self.split=="Val":
            class_index = idx
        elif self.split=="Test":
            class_index = idx
        id = self.identities[class_index]
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        
        label = self.annotations.loc[id]

        return image, torch.tensor(label)

# Combined Dataloader class implementation
class CombinedDataset(Dataset):
    def __init__(self, labels_file, root_dir, split, transform=None):
        self.annotations = pd.read_csv(labels_file, index_col=0)
        self.root_dir = root_dir
        self.split = split

        self.image_paths = []
        self.identities = self.annotations.axes[0].tolist()

        for id in self.identities:
            veridical_folder = id
            caricature_folder = id + "_caricature"

            veridical_images = os.listdir(os.path.join(root_dir, veridical_folder))
            caricature_images = os.listdir(os.path.join(root_dir, caricature_folder))

            random.seed(42)
            random.shuffle(veridical_images)
            random.shuffle(caricature_images)

            for img_index, img in enumerate(veridical_images):
                if split=="Train" and img_index<3:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))
                elif split=="Val" and img_index==3:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))
                elif split=="Test" and img_index==4:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))

            for img_index, img in enumerate(caricature_images):
                if split=="Train" and img_index<3:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))
                elif split=="Val" and img_index==3:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))
                elif split=="Test" and img_index==4:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if self.split == "Train":
            class_index = math.floor(idx / 6)
        elif self.split == "Val":
            class_index = math.floor(idx / 2)
        elif self.split == "Test":
            class_index = math.floor(idx / 2)

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

#Define the label file string, the number of features, and saved model string
label_string = ''
num_features = 0
model_load_string = "./saved_models/"

#Based on label type and dataset type, update these variables
if label_type == "prom":
    label_string = './labels/binary_prom_labels_extended.csv'
    num_features = 17

    if dataset_type == "veridical":
        model_load_string += "ver_"
    elif dataset_type == "caricature":
        model_load_string += "car_"
    elif dataset_type == "combined":
        model_load_string += "carver_"

    model_load_string += "prom_img.pt"

elif label_type == "sub":
    label_string = './labels/binary_sub_labels_extended.csv'
    num_features = 151

    if dataset_type == "veridical":
        model_load_string += "ver_"
    elif dataset_type == "caricature":
        model_load_string += "car_"
    elif dataset_type == "combined":
        model_load_string += "carver_"

    model_load_string += "sub_img.pt"

# Select dataset class based on dataset type
if dataset_type in ["caricature", "veridical"]:
    DatasetClass = CaricatureDataset
elif dataset_type == "combined":
    DatasetClass = CombinedDataset

#Create the datasets
# train_dataset = DatasetClass(labels_file=label_string, root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Train", transform=transform)
# val_dataset = DatasetClass(labels_file=label_string, root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Val", transform=transform)
test_dataset = DatasetClass(labels_file=label_string, root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Test", transform=transform)

#   Create the dataloaders
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Instatitate the model, loss function, and the optimizer
model = AlteredNet(num_features).to(device)
model.load_state_dict(torch.load(model_load_string))

total_correct = 0
total_samples = 0
all_predictions = []
all_labels = []

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device).float(), vlabels.to(device).float()
        voutputs = model(vinputs)

        voutputs = model.sigmoid_layer(voutputs)

        #Accuracy calculations
        vpredictions = np.round(voutputs.detach().cpu().numpy()).flatten()
        vlabels_np = vlabels.detach().cpu().numpy().flatten()

        # Collect predictions and labels for computing F1 score
        all_predictions.extend(vpredictions)
        all_labels.extend(vlabels_np)

        # Count correct predictions
        for index, prediction in enumerate(vpredictions):
            if prediction == vlabels_np[index]:
                total_correct += 1
            total_samples += 1

# Calculate accuracy
accuracy = total_correct / total_samples
print(f"Accuracy on the test set: {accuracy * 100:.1f}%")

# Calculate F1 score
f1 = f1_score(all_labels, all_predictions)
print(f"F1 score on the test set: {f1 * 100:.2f}%")