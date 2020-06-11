import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from functions import profile_sampling

data_dir = '/home/julia/data/gradients/'

# Load previously created profiles
with open(data_dir+'results/profiles/profiles.pkl', 'rb') as pf:
    profiles=pickle.load(pf)

# Load gradient image
data = nb.load(data_dir+'results/embedding/embed.nii.gz').get_data()

data_mesh = np.zeros((len(profiles), data.shape[3]))
for t in range(data.shape[3]):
    data_mesh[:,t] = np.squeeze(profile_sampling(data[:,:,:,t], profiles))

# load and save data on mesh
mesh = io.load_mesh_geometry(data_dir+'atlas/allen_api/brain_mesh.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh(data_dir+'results/embedding/embed_sampled_mesh.vtk', mesh)
