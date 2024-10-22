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
import wandb
from sklearn.model_selection import KFold

#Device declaration
device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training

#Define type of labels (prom/sub) and the type of dataset (caricature/veridical/combined)
label_type = "sub"
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

        self.model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_out_features)

    def forward(self, x):
        output = self.model(x)
        return output

""" class SimpleCNN(nn.Module):
    def __init__(self, num_features, dropout_rate=0.8):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Adjust the input size of the linear layer
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 54 * 44, num_features),
            nn.Dropout(dropout_rate)  # Add dropout layer
        )
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x """
    
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
model_save_string = "./saved_models/"
wandb_name = ""

#Based on label type and dataset type, update these variables
if label_type == "prom":
    label_string = './labels/binary_prom_labels_extended.csv'
    num_features = 17
    wandb_name += "prom "

    if dataset_type == "veridical":
        model_save_string += "ver_"
        wandb_name += "ver "
    elif dataset_type == "caricature":
        model_save_string += "car_"
        wandb_name += "car "
    elif dataset_type == "combined":
        model_save_string += "carver_"
        wandb_name += "carver "

    model_save_string += "prom_id_"

elif label_type == "sub":
    label_string = './labels/binary_sub_labels_extended.csv'
    num_features = 151
    wandb_name += "sub "

    if dataset_type == "veridical":
        model_save_string += "ver_"
        wandb_name += "ver "
    elif dataset_type == "caricature":
        model_save_string += "car_"
        wandb_name += "car "
    elif dataset_type == "combined":
        model_save_string += "carver_"
        wandb_name += "carver "

    model_save_string += "sub_id_"

# Select dataset class based on dataset type
if dataset_type in ["caricature", "veridical"]:
    DatasetClass = CaricatureDataset
elif dataset_type == "combined":
    DatasetClass = CombinedDataset

# Create the datasets
train_dataset = DatasetClass(labels_file=label_string, root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Train", transform=transform)
test_dataset = DatasetClass(labels_file=label_string, root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Test", transform=transform)

#Function that trains one epoch
def train_one_epoch(epoch_index, accp, totp):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair

        inputs, labels = data

        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(output, labels)

        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        #Apply sigmoid to the output
        output = model.sigmoid_layer(output)

        #Accuracy calculations
        predictions = np.round(output.detach().cpu().numpy()).flatten()
        target = labels.detach().cpu().numpy().flatten()
        for index, prediction in enumerate(predictions):
            if prediction == target[index]:
                accp += 1
            totp += 1

        torch.cuda.empty_cache()

    last_loss = running_loss/(i+1)
    acc = accp/totp

    return last_loss, acc

# Initialize variables to track the best validation loss and the corresponding model state
best_val_loss = float('inf')
best_model_state = None
best_model_fold = 0

# Initialize KFold with the desired number of splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate over the folds
for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
    print(f'Fold {fold + 1}')
    plot_name = wandb_name + "fold" + str(fold+1)

    #Initialize weights and biases
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        name=plot_name
    )

    #Instatitate the model, loss function, and the optimizer
    model = AlteredNet(num_features).to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    consecutive_no_improvement = 0
    max_consecutive_no_improvement = 10

    # Create data subsets for training and validation
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    # Create data loaders for training and validation
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    #Declaring the number of epochs
    EPOCHS = 250
    epoch_number = 0

    #Run training and/or validation for the set number of epochs
    for epoch in range(EPOCHS):
        accurate_predictions = 0
        total_predictions = 0

        vaccp = 0
        vtotp = 0
        vatl = np.array([])
        vapl = np.array([]) 

        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss, accuracy = train_one_epoch(epoch_number, accurate_predictions, total_predictions)

        # We don't need gradients on to do reporting
        model.eval()

        #Validation process
        with torch.no_grad():
            running_vloss = 0.0
            for i, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata

                vinputs, vlabels = vinputs.to(device).float(), vlabels.to(device).float()
                voutputs = model(vinputs)
                
                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

                voutputs = model.sigmoid_layer(voutputs)

                vpredictions = np.round(voutputs.detach().cpu().numpy()).flatten()
                #vpredictions = np.where(voutputs.detach().cpu().numpy() > 0.5, 1, 0)  # Apply thresholding for multi-label classificationv
                vtarget = vlabels.detach().cpu().numpy().flatten()
                for index, vprediction in enumerate(vpredictions):
                    if vprediction == vtarget[index]:
                        vaccp += 1
                    vtotp += 1

                vatl = np.append(vatl, vtarget)
                vapl = np.append(vapl, vpredictions)
                
        avg_vloss = running_vloss / (i+1)
        vacc = vaccp/vtotp

        f1 = f1_score(vatl, vapl, average='weighted')

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        print('ACC train {} valid {}'.format(accuracy, vacc))
        print('F1 score: {}'.format(f1))
        print()
        
        if avg_vloss < scheduler.best:
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= max_consecutive_no_improvement:
            print('Stopping early due to no improvement in validation loss.')
            break

        # Adjust learning rate based on validation loss
        scheduler.step(avg_vloss)

        # Inside the loop after evaluating the validation loss
        if avg_vloss < best_val_loss:
            best_val_loss = avg_vloss
            # Save the current model state as the best model state
            best_model_state = model.state_dict()
            best_model_fold = fold

        wandb.log({ 'Training' : avg_loss, 'Validation' : avg_vloss, 'Train Acc' : accuracy, 'Val Acc' : vacc, 'F1 Score': f1})

        torch.cuda.empty_cache()

        epoch_number += 1

    #Finish Weights and Biases
    wandb.finish()

model_save_string += "fold" + str(best_model_fold+1) + ".pt"
torch.save(best_model_state, model_save_string)