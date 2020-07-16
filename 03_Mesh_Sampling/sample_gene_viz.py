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

for p in [0, 1,3]:
    data[:,:,:,p][mask==0] = np.nan
    data_mesh = np.squeeze(profile_sampling(data[:,:,:,p], profiles))
    data_mesh_min = data_mesh.copy()
    data_mesh_min[np.isnan(data_mesh)] = np.nanmin(data_mesh)
    data_mesh_max = data_mesh.copy()
    data_mesh_max[np.isnan(data_mesh)] = np.nanmax(data_mesh)
    io.save_mesh('/home/julia/data/gradients/results/gene_expression/pc{}_viz_min.vtk'.format(p+1),
                 {'faces':mesh['faces'], 'points':mesh['points'],
                   'data':data_mesh_min})
    io.save_mesh('/home/julia/data/gradients/results/gene_expression/pc{}_viz_max.vtk'.format(p+1),
                 {'faces':mesh['faces'], 'points':mesh['points'],
                   'data':data_mesh_max})
