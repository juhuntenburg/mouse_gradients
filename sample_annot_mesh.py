import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from functions import profile_sampling


# Load previously created profiles
with open('/home/julia/data/gradients/results/profiles/profiles_highres.pkl', 'rb') as pf:
    profiles=pickle.load(pf)

data = nb.load('/home/julia/data/gradients/atlas/allen_api/regions/annot_finest.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight.nii.gz').get_data()
data[:,:,:][mask==0] = np.nan

data_mesh = np.squeeze(profile_sampling(data[:,:,:], profiles, method='winner'))

# load and save data on mesh
mesh = io.load_mesh_geometry('/home/julia/data/gradients/atlas/allen_api/brain_mesh.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh('/home/julia/data/gradients/atlas/allen_api/regions/annot_finest.vtk', mesh)
