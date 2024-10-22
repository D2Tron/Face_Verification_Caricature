#   Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import math
import random

#Device declaration
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

#Define type of labels (prom/sub) and the type of dataset (caricature/veridical/combined)
label_type = "prom"
dataset_type = "veridical"

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
model.load_state_dict(torch.load("./saved_models/ver_prom_id_fold2.pt"))

val_predictions_by_feature = [[] for _ in range(num_features)]
val_labels_by_feature = [[] for _ in range(num_features)]

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(test_loader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device).float(), vlabels.to(device).float()
        voutputs = model(vinputs)

        voutputs = model.sigmoid_layer(voutputs)

        #Update the lists for the val sets
        for feature_index in range(num_features):
            val_predictions_by_feature[feature_index].extend(np.round(voutputs[:, feature_index].detach().cpu().numpy()))
            val_labels_by_feature[feature_index].extend(vlabels[:, feature_index].detach().cpu().numpy())

# Feature names list
feature_names = ["cheekbones", "cheeks", "chin", "ears", "eyebrows", "eyelids", "eyes", "facial_hair", "forehead", "hair", "head", "lips", "mouth", "neck", "nose", "skin", "teeth"]

""" feature_names = ['cheekbones_high', 'cheekbones_sharp', 
                 'cheeks_chubby/full', 'cheeks_dimples', 'cheeks_thin/hollow', 
                 'chin_cleft', 'chin_crooked', 'chin_double chin', 'chin_forward', 'chin_pointed', 'chin_rounded', 'chin_scar', 'chin_square', 'chin_strong jawline', 'chin_weak jawline', 
                 'ears_big', 'ears_flat', 'ears_high', 'ears_low', 'ears_pierced', 'ears_pointy', 'ears_small', 'ears_stick out', 
                 'eyebrows_arched (v-shaped)', 'eyebrows_bushy', 'eyebrows_curved down', 'eyebrows_far apart', 'eyebrows_flat', 'eyebrows_furrowed', 'eyebrows_light', 'eyebrows_long', 'eyebrows_scar', 'eyebrows_short', 'eyebrows_slanted down', 'eyebrows_thick', 'eyebrows_thin', 'eyebrows_unibrow', 
                 'eyelids_drooping', 'eyelids_hooded', 'eyelids_puffy', 'eyelids_receded', 
                 'eyes_almond', 'eyes_bags under eyes', 'eyes_crows feet', 'eyes_deep-set', 'eyes_glasses', 'eyes_lazy eye', 'eyes_light-colored', 'eyes_long eyelashes', 'eyes_narrow', 'eyes_narrow-set', 'eyes_round', 'eyes_slanted down', 'eyes_slanted up', 'eyes_small', 'eyes_stick out', 'eyes_wide', 'eyes_wide-set', 'eyes_wide-x', 
                 'facial hair_beard', 'facial hair_goatee', 'facial hair_handlebar', 'facial hair_messy', 'facial hair_mustache', 'facial hair_sideburns', 'facial hair_soul patch', 'facial hair_stubble', 'facial hair_thick', 'facial hair_thin', 'facial hair_trimmed', 'facial hair_white', 
                 'forehead_big', 'forehead_narrow', 'forehead_scar', 'forehead_small', 'forehead_wide', 'forehead_wrinkled', 
                 'hair_bald', 'hair_bangs', 'hair_big', 'hair_black', 'hair_blond', 'hair_curly', 'hair_dreads', 'hair_hat', 'hair_long', 'hair_receding hairline', 'hair_red', 'hair_short', 'hair_slicked back', 'hair_white', 'hair_white streaks', 'hair_widows peak', 
                 'head_big', 'head_long', 'head_round', 'head_small', 'head_square', 'head_wide', 
                 'lips_downturned', 'lips_large', 'lips_medial cleft', 'lips_pouty/full', 'lips_red lipstick', 'lips_thick lower', 'lips_thin', 'lips_thin upper', 'lips_upturned',  
                 'mouth_big/wide', 'mouth_crooked', 'mouth_small', 
                 "neck_Adam's apple", 'neck_lines', 'neck_tattoos', 'neck_thick', 
                 'nose_bulbous', 'nose_button', 'nose_cleft', 'nose_crooked', 'nose_dorsal hump', 'nose_flared nostrils', 'nose_flat', 'nose_hooked', 'nose_long', 'nose_pointy', 'nose_rounded tip', 'nose_short', 'nose_small', 'nose_small nostrils', 'nose_thin', 'nose_thin bridge', 'nose_upturned', 'nose_v-shaped', 'nose_well-defined tip', 'nose_wide', 'nose_wide bridge', 'nose_wide nostrils', 'nose_wide tip', 
                 'skin_freckles', 'skin_mole', 'skin_pale', 'skin_rough', 'skin_smooth', 
                 'teeth_big', 'teeth_buck', 'teeth_crooked', 'teeth_gap', 'teeth_overbite', 'teeth_small', 'teeth_straight', 'teeth_white'] """

# Create a dictionary mapping feature indices to names
feature_dict = dict(enumerate(feature_names))

# Count of each feature across all identities
feature_counts = [0] * num_features

# F1 scores for validation
f1_val_by_feature = []
for feature_index in range(num_features):
    val_predictions_array = np.array(val_predictions_by_feature[feature_index])
    val_labels_array = np.array(val_labels_by_feature[feature_index])

    # f1_val = f1_score(val_labels_array, val_predictions_array, average='binary')
    f1_val = accuracy_score(val_labels_array, val_predictions_array)
    f1_val_by_feature.append(f1_val)

    # Count the number of identities with this feature
    feature_counts[feature_index] += np.sum(val_labels_array)

feature_counts = [count // 5 for count in feature_counts]

# Sort F1 scores and corresponding feature names based on the counts
sorted_f1_val_by_counts = sorted(zip(f1_val_by_feature, feature_names, feature_counts), key=lambda x: x[2], reverse=True)

# Export sorted F1 scores and counts to a CSV file
columns = ['F1 Score', 'Feature Name', 'Frequency Count']
df_val = pd.DataFrame(sorted_f1_val_by_counts, columns=columns)

#df_train.to_csv('./saved_lists/f1_train_carver_prom.csv', index=False)
df_val.to_csv('./saved_lists/f1_lists/acc_ver_prom_id.csv', index=False)

# Print or use the sorted results
counter = 0
print("\nSorted F1 Scores by Feature Counts (Validation):")
for f1_score, feature_name, count in sorted_f1_val_by_counts:
    if counter == 20:
        break
    print(f"{feature_name}: {f1_score} (Count: {count})")
    counter += 1

print("\nFull list of feature counts")
for _, fname, cnt in sorted_f1_val_by_counts:
    print(f"{fname}: {int(cnt)}")

from collections import Counter
feature_counts.sort()
count_of_counts = Counter(feature_counts)

print("\nCount of Counts:")
for count, frequency in count_of_counts.items():
    print(f"{frequency} features present {int(count)} times")