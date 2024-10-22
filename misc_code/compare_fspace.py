import numpy as np
import torch
import torch.nn.functional as F

ver_fspace = np.load('outputs_ver_prom.npy')
car_fspace = np.load('outputs_car_prom.npy')
static_fspace = np.load('fspace_Static.npy')

# print(ver_fspace[0][:10])
# ver_fspace = np.flip(ver_fspace)
# print(ver_fspace[0][:10])
ver_tensor = torch.from_numpy(ver_fspace.copy())
car_tensor = torch.from_numpy(car_fspace.copy())
static_tensor = torch.from_numpy(static_fspace)

#Calculate cosine similarity
cosine_similarities = F.cosine_similarity(ver_tensor, car_tensor, dim=-1)
# euc_dist = torch.sqrt(torch.sum(torch.pow(torch.subtract(ver_tensor, car_tensor), 2), dim=1)) 

#Print or use the cosine similarities as needed
print("Cosine Similarities:", cosine_similarities)

# from sklearn.metrics.pairwise import cosine_similarity

# randomvec1 = np.concatenate(([1], [0] * 511))
# randomvec2 = np.random.rand(512)

# randomvec1 = randomvec1.reshape(1, -1)
# randomvec2 = randomvec2.reshape(1, -1)

# cossim = cosine_similarity(randomvec1, randomvec2)
# print("cos sim:", cossim)
