#   Imports
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms, utils, datasets, models
from torchvision.io import read_image
from sklearn.metrics import multilabel_confusion_matrix, f1_score, ConfusionMatrixDisplay, confusion_matrix
import math
import random
import wandb
from matplotlib import pyplot as plt

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

    def forward(self, x):
        output = self.model(x)
        return output

#Dataloader class implementation
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

            random.shuffle(veridical_images)
            random.shuffle(caricature_images)

            for img_index, img in enumerate(veridical_images):
                if split == "Train" and img_index < 4:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))
                elif split == "Test" and img_index == 4:
                    self.image_paths.append((os.path.join(root_dir, veridical_folder, img)))

            for img_index, img in enumerate(caricature_images):
                if split == "Train" and img_index < 4:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))
                elif split == "Test" and img_index == 4:
                    self.image_paths.append((os.path.join(root_dir, caricature_folder, img)))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if self.split == "Train":
            class_index = math.floor(idx / 8)
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

#Create the datasets
train_dataset = CombinedDataset(labels_file='binary_sub_labels_extended.csv', root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Train", transform=transform)
test_dataset = CombinedDataset(labels_file='binary_sub_labels_extended.csv', root_dir='/home/jsutariya/Desktop/Project/ourcar/', split="Test", transform=transform)

#   Create the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#Initialize weights and biases
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    name="Esub carver schdlr p5"
)

#Instatitate the model, loss function, and the optimizer
model = AlteredNet(151).to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
consecutive_no_improvement = 0
max_consecutive_no_improvement = 10

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

        output = model.sigmoid_layer(output)

        #Accuracy calculations
        predictions = np.round(output.detach().cpu().numpy()).flatten()
        target = labels.detach().cpu().numpy().flatten()
        for index, prediction in enumerate(predictions):
            if prediction == target[index]:
                accp += 1
            totp += 1

        torch.cuda.empty_cache()

    last_loss = running_loss/len(train_loader.dataset)
    acc = accp/totp

    return last_loss, acc

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
    # model.train(True)
    model.train()
    avg_loss, accuracy = train_one_epoch(epoch_number, accurate_predictions, total_predictions)

    # We don't need gradients on to do reporting
    # model.train(False)
    model.eval()

    #Validation process
    with torch.no_grad():
        running_vloss = 0.0
        for i, vdata in enumerate(test_loader):
            vinputs, vlabels = vdata

            vinputs, vlabels = vinputs.to(device).float(), vlabels.to(device).float()
            voutputs = model(vinputs)
            
            vloss = loss_fn(voutputs, vlabels)

            running_vloss += vloss.item()

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
            
    avg_vloss = running_vloss / len(test_loader.dataset)

    vacc = vaccp/vtotp

    f1 = f1_score(vatl, vapl, average='weighted')

    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    print('ACC train {} valid {}'.format(accuracy, vacc))
    print('F1 score: {}'.format(f1))
    print()
    
    wandb.log({ 'Training' : avg_loss, 'Validation' : avg_vloss, 'Train Acc' : accuracy, 'Val Acc' : vacc, 'F1 Score': f1})

    if avg_vloss < scheduler.best:
        consecutive_no_improvement = 0
    else:
        consecutive_no_improvement += 1

    if consecutive_no_improvement >= max_consecutive_no_improvement:
        print('Stopping early due to no improvement in validation loss.')
        break

    # Adjust learning rate based on validation loss
    scheduler.step(avg_vloss)

    torch.cuda.empty_cache()

    epoch_number += 1

#Finish Weights and Biases
wandb.finish()

torch.save(model.state_dict(), "./carver_Esub.pt")