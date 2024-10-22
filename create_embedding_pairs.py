import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision import transforms

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
    
# Define a custom dataset to load verification pairs
class VerificationPairsDataset(Dataset):
    def __init__(self, pairs_df, transform=None):
        self.pairs_df = pairs_df
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        pair = self.pairs_df.iloc[idx]
        image1_path = pair['ver_image_path']
        image2_path = pair['car_image_path']
        label = pair['label']

        # Load and transform images if needed
        # Modify this part based on how you load and transform images in your dataset
        image1 = torchvision.io.read_image(image1_path, mode=torchvision.io.ImageReadMode.RGB)
        image2 = torchvision.io.read_image(image2_path, mode=torchvision.io.ImageReadMode.RGB)

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transofrm(image2)

        return image1, image2, label

#Transformer
transform = transforms.Compose([
    transforms.Resize((218, 178), antialias=True)
])

# Read in the verification pairs csv to a dataframe
verification_pairs_df = pd.read_csv('./saved_lists/verification_pairs_id.csv')

# Initialize a DataLoader for verification pairs
verification_pairs_dataset = VerificationPairsDataset(verification_pairs_df)
verification_pairs_loader = DataLoader(verification_pairs_dataset, batch_size=64, shuffle=False)

# Instatitate the model, loss function, and the optimizer
car_model = AlteredNet(151).to(device)
ver_model = AlteredNet(151).to(device)

# Initialize the model with the best model state
car_model.load_state_dict(torch.load('./saved_models/car_sub_id_fold1.pt'))
ver_model.load_state_dict(torch.load('./saved_models/ver_sub_id_fold3.pt'))

# car_model.load_state_dict(torch.load('./saved_models/car_prom_id_fold1.pt'))
# ver_model.load_state_dict(torch.load('./saved_models/ver_prom_id_fold2.pt'))

# car_model.load_state_dict(torch.load('./saved_models/car_sub_img.pt'))
# ver_model.load_state_dict(torch.load('./saved_models/ver_sub_img.pt'))

# car_model.load_state_dict(torch.load('./saved_models/car_sub_img.pt'))
# ver_model.load_state_dict(torch.load('./saved_models/ver_sub_img.pt'))

# Remove last layer
# car_model.remove_last_layer()
# ver_model.remove_last_layer()

car_model.eval()
ver_model.eval()

# Create lists to store the embeddings and labels
image1_embeddings_list = []
image2_embeddings_list = []
labels_list = []

# Loop through the verification pairs to compute embeddings
with torch.no_grad():
    for image1, image2, labels in verification_pairs_loader:
        image1 = image1.to(device).float()
        image2 = image2.to(device).float()
        
        # Forward pass to obtain the embeddings
        embeddings1 = ver_model(image1)
        embeddings2 = car_model(image2)

        # Apply sigmoid layer
        # embeddings1 = ver_model.sigmoid_layer(embeddings1)
        # embeddings2 = car_model.sigmoid_layer(embeddings2)

        # Convert ito a numpy array
        embeddings1 = embeddings1.detach().cpu().numpy()
        embeddings2 = embeddings2.detach().cpu().numpy()
        
        # Squeeze output to one dimension
        while len(embeddings1.shape) > 2:
            embeddings1 = embeddings1.squeeze(-1)
        while len(embeddings2.shape) > 2:
            embeddings2 = embeddings2.squeeze(-1)

        # Append the embeddings and labels to the lists
        image1_embeddings_list.append(embeddings1)
        image2_embeddings_list.append(embeddings2)
        labels_list.append(labels.numpy())

# Concatenate embeddings and labels
image1_embeddings = np.concatenate(image1_embeddings_list, axis=0)
image2_embeddings = np.concatenate(image2_embeddings_list, axis=0)
labels = np.concatenate(labels_list, axis=0)

# Create a DataFrame
df = pd.DataFrame({'embedding1': image1_embeddings.tolist(),
                   'embedding2': image2_embeddings.tolist(),
                   'label': labels.tolist()})

# Save DataFrame to a CSV file
csv_file = './saved_lists/verification_embeddings_id_sub.csv'
df.to_csv(csv_file, index=False)