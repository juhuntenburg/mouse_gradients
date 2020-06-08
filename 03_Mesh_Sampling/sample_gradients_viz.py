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
mesh = io.load_mesh('/home/julia/data/gradients/atlas/allen_api/annotation/annot.vtk')


for d in ['ad2', 'ad3', 'csd']:
    # Load gradient image
    data = nb.load('/home/julia/data/gradients/results/repro/%s_embed.nii.gz' %d).get_data()
    data_mesh = np.zeros((len(profiles),))

    for t in range(6):
        data_mesh = np.squeeze(profile_sampling(data[:,:,:,t], profiles))
        data_mesh_min = data_mesh.copy()
        data_mesh_min[np.isnan(data_mesh)] = np.nanmin(data_mesh)
        data_mesh_max = data_mesh.copy()
        data_mesh_max[np.isnan(data_mesh)] = np.nanmax(data_mesh)
        io.save_mesh('/home/julia/data/gradients/results/repro/%s_embed_viz_g%i_min.vtk'%(d,t),
                     {'faces':mesh['faces'], 'points':mesh['points'],
                       'data':data_mesh_min})
        io.save_mesh('/home/julia/data/gradients/results/repro/%s_embed_viz_g%i_max.vtk'%(d,t),
                     {'faces':mesh['faces'], 'points':mesh['points'],
                       'data':data_mesh_max})
