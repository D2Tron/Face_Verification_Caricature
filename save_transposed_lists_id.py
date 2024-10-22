#   Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.metrics import multilabel_confusion_matrix
import math
import random

#Device declaration
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

#Define type of labels (prom/sub) and the type of dataset (caricature/veridical/combined)
label_type = "sub"
dataset_type = "combined"

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

        random.seed(42)
        random.shuffle(self.identities)

        num_identities = len(self.identities)
        train_size = num_identities - 24  # Exclude 24 identities for the test set

        # Adjust the splitting based on the "split" parameter
        if split == "Train":
            self.identities = self.identities[:train_size]
        elif split == "Test":
            self.identities = self.identities[train_size:]

        for id in self.identities:
            folder_name = id + folder_suffix

            image_names = os.listdir(os.path.join(root_dir, folder_name))
            random.shuffle(image_names)
            for img_index, img in enumerate(image_names):
                if img_index < 5:
                    self.image_paths.append((os.path.join(root_dir, folder_name, img)))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        class_index = math.floor(idx / 5)
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

        random.seed(42)
        random.shuffle(self.identities)

        num_identities = len(self.identities)
        train_size = num_identities - 24  # Exclude 24 identities for the test set

        # Adjust the splitting based on the "split" parameter
        if split == "Train":
            self.identities = self.identities[:train_size]
        elif split == "Test":
            self.identities = self.identities[train_size:]

        for id in self.identities:
            veridical_folder = id
            caricature_folder = id + "_caricature"

            veridical_images = os.listdir(os.path.join(root_dir, veridical_folder))
            caricature_images = os.listdir(os.path.join(root_dir, caricature_folder))

            random.shuffle(veridical_images)
            random.shuffle(caricature_images)

            for img_index, img in enumerate(veridical_images):
                if img_index < 5:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))

            for img_index, img in enumerate(caricature_images):
                if img_index < 5:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        class_index = math.floor(idx / 10)
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

#Based on label type and dataset type, update these variables
if label_type == "prom":
    label_string = './labels/binary_prom_labels_extended.csv'
    num_features = 17

elif label_type == "sub":
    label_string = './labels/binary_sub_labels_extended.csv'
    num_features = 151

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
model.load_state_dict(torch.load("./saved_models/carver_sub_id_fold1.pt"))

all_true_labels = np.array([])
all_predictions = np.array([])
all_true_labels_itemized = np.zeros([0, num_features])
all_predictions_itemized = np.zeros([0, num_features])

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device).float(), vlabels.to(device).float()
        voutputs = model(vinputs)

        voutputs = model.sigmoid_layer(voutputs)

        predictions = np.where(voutputs.detach().cpu().numpy() > 0.5, 1, 0)  # Apply thresholding for multi-label classification
        vlabels = vlabels.detach().cpu().numpy()
        
        all_true_labels = np.append(all_true_labels, vlabels)
        all_predictions = np.append(all_predictions, predictions)
        
        predictions = predictions.transpose()

        #Get itemized labels and predictions
        all_true_labels_itemized = np.concatenate((all_true_labels_itemized, vlabels), axis=0)
        all_predictions_itemized = np.concatenate((all_predictions_itemized, predictions.T), axis=0)

print('Confusion Matrix: ', multilabel_confusion_matrix(all_true_labels, all_predictions))

#Transpose the labels and predictions
transposed_labels = all_true_labels_itemized.transpose()
transposed_predictions = all_predictions_itemized.transpose()

np.save('./saved_lists/transposed_lists/tlabels_sub_carver_id.npy', transposed_labels)
np.save('./saved_lists/transposed_lists/tpredictions_sub_carver_id.npy', transposed_predictions)