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

data = nb.load('/home/julia/data/gradients/genes/pca_img.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/genes/gene_mask.nii.gz').get_data()

data_mesh = np.zeros((len(profiles), 5))
# Load gradient image
for p in range(5):
    data[:,:,:,p][mask==0] = np.nan
    data_mesh[:,p] = np.squeeze(profile_sampling(data[:,:,:,p], profiles))

# load and save data on mesh
mesh = io.load_mesh_geometry('/home/julia/data/gradients/atlas/allen_api/brain_mesh.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh('/home/julia/data/gradients/genes/pca_sampled_mesh.vtk', mesh)
