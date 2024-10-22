import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

prom_labels = pd.read_csv('binary_prom_labels.csv', index_col=0)
sub_labels = pd.read_csv('binary_sub_labels.csv', index_col=0)

identity1 = np.array(prom_labels.iloc[14])
identity2 = np.array(prom_labels.iloc[15])

labels1 = torch.from_numpy(identity1.copy())
labels2 = torch.from_numpy(identity2.copy())

#Calculate cosine similarity
cosine_similarities = F.cosine_similarity(labels1, labels2, dim=-1)
# euc_dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(ver_tensor, car_tensor), 2), dim=1)) 

#Print or use the cosine similarities as needed
print("Cosine Similarities:", cosine_similarities)