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

for d in ['piri', 'hpc', 'comb', 'hpc_nonzero', 'piri_nonzero']:
    # Load gradient image
    data = nb.load('/home/julia/data/gradients/results/distance/euclidian_%s.nii.gz' %d).get_data()
    data_mesh = np.squeeze(profile_sampling(data, profiles))

    # load and save data on mesh
    mesh = io.load_mesh('/home/julia/data/gradients/atlas/allen_api/regions/annot_finest.vtk')
    mesh['data'] = np.nan_to_num(data_mesh)
    io.save_mesh('/home/julia/data/gradients/results/distance/euclidian_%s_mesh.vtk' %d, mesh)
