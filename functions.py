import numpy as np
from nighres import io
import nibabel as nb
import pickle
from glob import glob
import os


def find_profiles(mesh, normals, mask, resolution=200):

    mask=np.array(mask, dtype=np.float32)
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
                    if len(voxels) < 20:
                        voxels.append(voxel)
                    # if the nan occurs later, stop sampling
                    else:
                        break
                else:
                    # the actual profile only gets appended for none nan voxels
                    voxels.append(voxel)
                    profile.append(voxel)

                # don't go deeper than 11 voxels (2.2 mm)
                if len(profile) > 80:
                    break

        # remove the first voxel to avoid partial voluming
        profiles.append(profile[1:])

    return profiles



def profile_sampling(data, profiles, method='mean'):

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
            if method == 'mean':
                mesh_data[p,:] = np.nanmean(point_data, axis=0)
            elif method == 'winner':
                if len(np.squeeze(point_data).shape) > 1:
                    'winner method only available for 1D data'
                    break
                else:
                    labels = np.unique(point_data)
                    lens = []
                    for u in labels:
                        lens.append(np.where(point_data==u)[0].shape[0])
                    mesh_data[p,:] = labels[np.where(lens == np.max(lens))[0][0]]

    return mesh_data
