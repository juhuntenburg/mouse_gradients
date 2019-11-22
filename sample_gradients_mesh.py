import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from functions import profile_sampling


# Load previously created profiles
with open('/home/julia/data/gradients/results/profiles/profiles.pkl', 'rb') as pf:
    profiles=pickle.load(pf)

# Load gradient image
data = nb.load('/home/julia/data/gradients/results/embedding_vol/embed.nii.gz').get_data()

data_mesh = np.zeros((len(profiles), data.shape[3]))
for t in range(data.shape[3]):
    data_mesh[:,t] = np.squeeze(profile_sampling(data[:,:,:,t], profiles))

# load and save data on mesh
mesh = io.load_mesh_geometry('/home/julia/data/gradients/atlas/allen_api/brain_mesh.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh('/home/julia/data/gradients/results/embedding_vol/embed_sampled_mesh.vtk', mesh)
