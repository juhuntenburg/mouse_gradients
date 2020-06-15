import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from nipype.interfaces import fsl
from profile_functions import profile_sampling

data_dir = '/home/julia/data/gradients/'
for origin in ['hippocampus', 'piriform']:

    # Dilate mask
    dilate = fsl.maths.MathsCommand(in_file=data_dir+'allen_atlas/%s_200um.nii.gz' %origin,
                                    out_file=data_dir+'results/origins/%s_200um_dil.nii.gz' %origin,
                                    args='-dilM')
    dilate.run()

    #Load previously created profiles
    with open(data_dir+'results/profiles/profiles.pkl', 'rb') as pf:
        profiles=pickle.load(pf)

    data = nb.load(data_dir+'results/origins/%s_200um_dil.nii.gz' %origin).get_data()
    mask = nb.load(data_dir+'allen_atlas/cortex_mask_tight_200um.nii.gz').get_data()
    data[:,:,:][mask==0] = np.nan

    data_mesh = np.squeeze(profile_sampling(data[:,:,:], profiles))

    # load and save data on mesh
    mesh = io.load_mesh(data_dir+'allen_atlas/brain_mesh.vtk')
    mesh['data'] = np.nan_to_num(data_mesh)
    io.save_mesh(data_dir+'results/origins/%s_sampled_mesh.vtk' %origin, mesh)
