import numpy as np
from nighres import io
import nibabel as nb
import pickle
from glob import glob
import os
from profile_functions import find_profiles, profile_sampling

data_dir = '/home/julia/data/gradients/'
mesh = io.load_mesh_geometry(data_dir+'allen_atlas/brain_mesh.vtk')
normals = np.load(data_dir+'allen_atlas/brain_mesh_normals.npy')
# Brain mesh normals created by loading mesh into paraview, using filter 'GenerateSurfaceNormals',
# saving the result as csv and then converting the normals to npy in Python
mask = nb.load(data_dir+'allen_atlas/cortex_mask_tight.nii.gz').get_data()

# Create the profiles
profiles = find_profiles(mesh, normals, mask, resolution=25, highres=True)
with open(data_dir+'results/profiles/profiles_highres.pkl', 'wb') as f:
    pickle.dump(profiles, f)
