import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from profile_functions import profile_sampling

# Load previously created profiles
with open('/home/julia/data/gradients/results/profiles/profiles.pkl', 'rb') as pf:
    profiles=pickle.load(pf)
mesh = io.load_mesh('/home/julia/data/gradients/allen_atlas/brain_mesh.vtk')



data = nb.load('/home/julia/data/gradients/results/gene_expression/pca_img.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/allen_atlas/gene_expression/gene_mask.nii.gz').get_data()

data_mesh = np.zeros((len(profiles), data.shape[3]))

for p in range(4):
    data[:,:,:,p][mask==0] = np.nan
    data_mesh[:,p] = np.squeeze(profile_sampling(data[:,:,:,p], profiles))

io.save_mesh('/home/julia/data/gradients/results/gene_expression/pcs_mesh.vtk',
             {'faces':mesh['faces'], 'points':mesh['points'],'data':np.nan_to_num(data_mesh)})
