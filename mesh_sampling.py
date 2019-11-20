import numpy as np
from nighres import io
import nibabel as nb
import pickle
from glob import glob
import os


def find_profiles(mesh, normals, mask, resolution=200):

    mask[mask==0]=np.nan
    profiles = []

    for p in range(len(mesh["points"])):
        voxels = []
        profile = []
        first_round = True
        point = mesh["points"][p]

        while True:
            if first_round is True:
                voxel = np.array(np.round(point/resolution), dtype="int16")
                same_voxel = False
                first_round = False
            else:
                # move forward
                point = point - normals[p]
                voxel = np.array(np.round(point/resolution), dtype="int16")
                # check if this is a new voxel
                same_voxel = np.all(voxels[-1]==voxel)

            if same_voxel == False:
                # check that voxel is still in image
                try:
                    value = mask[voxel[0], voxel[1], voxel[2]]
                except IndexError:
                    break
                # when outside the cortex mask
                if np.isnan(mask[voxel[0], voxel[1], voxel[2]]) == True:
                    # allow to go on for a few voxels in the beginning
                    if len(voxels) < 2:
                        voxels.append(voxel)
                    # if the nan occurs later, stop sampling
                    else:
                        break
                else:
                    # the actual profile only gets appended for none nan voxels
                    voxels.append(voxel)
                    profile.append(voxel)

                if len(profile) > 20:
                    print(p, " over 20")
                    break

        profiles.append(profile)

    return profiles



def profile_sampling(data, profiles):

    if len(data.shape) == 3:
        data = data[:,:,:,np.newaxis]

    mesh_data = np.zeros((len(profiles), data.shape[3]))

    for p in range(len(profiles)):
        if len(profiles[p]) == 0:
            mesh_data[p,:] = np.nan
        else:
            point_data = np.zeros((len(profiles[p]), data.shape[3]))
            for v in range(len(profiles[p])):
                vox = profiles[p][v]
                point_data[v,:] = data[vox[0], vox[1], vox[2], :]
            mesh_data[p,:] = np.nanmean(point_data, axis=0)

    return mesh_data



mesh = io.load_mesh_geometry("/home/julia/data/gradients/atlas/allen_api/brain_mesh.vtk")
normals = np.load("/home/julia/data/gradients/atlas/allen_api/brain_mesh_normals.npy")
mask = nb.load("/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight_200um.nii.gz").get_data()

profiles = find_profiles(mesh, normals, mask, resolution=200)
with open('/home/julia/data/gradients/results/mesh_sampling/profiles.pkl', 'wb') as f:
    pickle.dump(profiles, f)

# For QA
lvl = nb.load("/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight_lvl_200um.nii.gz").get_data()
mesh_data = profile_sampling(lvl, profiles)
mesh_data = np.nan_to_num(mesh_data)
mesh['data']=mesh_data
io.save_mesh('/home/julia/data/gradients/results/mesh_sampling/lvl_qa_mesh_sampling.vtk', mesh)

profile_length = []
for p in range(len(mesh["points"])):
    profile_length.append(len(profiles[p]))
profile_length=np.asarray(profile_length)
mesh['data'] = profile_length
io.save_mesh('/home/julia/data/gradients/results/mesh_sampling/profile_length_mesh_sampling.vtk', mesh)
