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

data = nb.load('/home/julia/data/gradients/atlas/epfl_cell_atlas/cortex_cells_sig_200um_smooth045.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight_200um.nii.gz').get_data()
data[:,:,:][mask==0] = np.nan

data_mesh = np.squeeze(profile_sampling(data[:,:,:], profiles))

# load and save data on mesh
mesh = io.load_mesh('/home/julia/data/gradients/atlas/allen_api/regions/annot_finest.vtk')
mesh['data'] = np.nan_to_num(data_mesh)
io.save_mesh('/home/julia/data/gradients/results/cells/cells_sig_sampled_mesh.vtk', mesh)
