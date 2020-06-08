import numpy as np
from nighres import io
import nibabel as nb
import pickle
from glob import glob
import os
from profile_functions import find_profiles, profile_sampling


mesh = io.load_mesh_geometry("/home/julia/data/gradients/allen_atlas/brain_mesh.vtk")
normals = np.load("/home/julia/data/gradients/allen_atlas/brain_mesh_normals.npy")
# Brain mesh normals created by loading mesh into paraview, using filter "GenerateSurfaceNormals",
# saving the result as csv and then converting the normals to npy
mask = nb.load("/home/julia/data/gradients/allen_atlas/cortex_mask_tight_200um.nii.gz").get_data()

# Create the profiles
profiles = find_profiles(mesh, normals, mask, resolution=200)
with open('/home/julia/data/gradients/results/profiles/profiles.pkl', 'wb') as f:
    pickle.dump(profiles, f)

# Calculate profile length for QA
profile_length = []
for p in range(len(mesh["points"])):
    profile_length.append(len(profiles[p]))
profile_length=np.asarray(profile_length)
mesh['data'] = profile_length
io.save_mesh('/home/julia/data/gradients/results/profiles/profile_length_mesh_sampling.vtk', mesh)

# Sample levelset for QA
lvl = nb.load("/home/julia/data/gradients/allen_atlas/cortex_mask_tight_lvl_200um.nii.gz").get_data()
mesh_data = profile_sampling(lvl, profiles)
mesh_data = np.nan_to_num(mesh_data)
mesh['data']=mesh_data
io.save_mesh('/home/julia/data/gradients/results/profiles/lvl_qa_mesh_sampling.vtk', mesh)
