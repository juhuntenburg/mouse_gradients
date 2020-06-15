import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from profile_functions import profile_sampling

data_dir = '/home/julia/data/gradients/'

# Load previously created profiles
with open(data_dir+'results/profiles/profiles_highres.pkl', 'rb') as pf:
    profiles=pickle.load(pf)

# Load gradient image
data = nb.load(data_dir+'allen_atlas/annot_finest.nii.gz').get_data()
mask = nb.load(data_dir+'allen_atlas/cortex_mask_tight.nii.gz').get_data()

data[:,:,:][mask==0] = np.nan
data_mesh = np.squeeze(profile_sampling(data[:,:,:], profiles, method='winner'))

# load and save data on mesh
mesh = io.load_mesh_geometry(data_dir+'allen_atlas/brain_mesh.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh(data_dir+'allen_atlas/annot_finest_sampled_mesh.vtk', mesh)
