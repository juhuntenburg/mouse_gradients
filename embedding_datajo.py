from __future__ import division
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb
import hcp_corr


ne.set_num_threads(ne.ncores-1)


rest_file = '/home/julia/projects/gradients/data_jo/%s_%s_rest%s_smooth_3.npy'
mask_file = '/home/julia/projects/gradients/data_jo/'
corr_file = '/home/julia/projects/gradients/data_jo/corr.hdf5'
embed_file = '/home/julia/projects/gradients/data_jo/embed.npy'
embed_img = '/home/julia/projects/gradients/data_jo/embed.nii.gz'
embed_dict_file = '/home/julia/projects/gradients/data_jo/embed_dict.pkl'

calc_corr = False
save_corr = False
calc_embed = True

subjects = []
sessions = []


def avg_correlation(ts_files, thr=None):
    '''
    Calculates average connectivity matrix using hcp_corr package for memory
    optimization: https://github.com/NeuroanatomyAndConnectivity/hcp_corr
    '''
    # make empty avg corr matrix
    if type(ts_files[0]) == str:
        get_size = np.load(ts_files[0]).shape[0]
    elif type(ts_files[0]) == np.ndarray:
        get_size = ts_files[0].shape[0]

    full_shape = (get_size, get_size)
    if np.mod((get_size**2-get_size), 2) == 0.0:
        avg_corr = np.zeros((get_size**2-get_size)/2)
    else:
        print('size calculation no zero mod')

    count = 0
    for rest in ts_files:
        # load time series
        if type(rest) == str:
            rest = np.load(rest)
        elif type(rest) == np.ndarray:
            pass
        # calculate correlations matrix
        print('...corrcoef')
        corr = hcp_corr.corrcoef_upper(rest)
        del rest
        # threshold / transform
        if thr is None:
            # r-to-z transform and add to avg
            print('...transform')
            avg_corr += ne.evaluate('arctanh(corr)')
        else:
            # threshold and add to avg
            print('...threshold')
            thr = np.percentile(corr, 100-thr)
            avg_corr[np.where(corr > thr)] += 1
        del corr
        count += 1
    # divide by number of sessions included
    print('...divide')
    avg_corr /= count
    # transform back if necessary
    if thr is None:
        print('...back transform')
        avg_corr = np.nan_to_num(ne.evaluate('tanh(avg_corr)'))

    return avg_corr, full_shape


def recort(n_vertices, data, cortex, increase):
    '''
    Helper function to rewrite masked, embedded data to full cortex
    (copied from Daniel Margulies)
    '''
    d = np.zeros(n_vertices)
    count = 0
    for i in cortex:
        d[i] = data[count] + increase
        count = count + 1
    return d


def embedding(upper_corr, full_shape, mask, n_components):
    '''
    Diffusion embedding on connectivity matrix using mapaling package:
    https://github.com/satra/mapalign
    '''
    # reconstruct full matrix
    print('...full matrix')
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    full_corr += full_corr.T

    # apply mask
    print('...mask')
    masked_corr = np.delete(full_corr, mask, 0)
    del full_corr
    masked_corr = np.delete(masked_corr, mask, 1)

    mask_flat = mask.flatten()
    all_voxel = range(mask_flat.shape[0])
    brain = np.delete(all_voxel, np.where(mask_flat == 0)[0])

    # run actual embedding
    print('...embed')
    K = (masked_corr + 1) / 2.
    del masked_corr
    K[np.where(np.eye(K.shape[0]) == 1)] = 1.0

    embedding_results, \
        embedding_dict = embed.compute_diffusion_map(K,
                                                     n_components=n_components,
                                                     overwrite=True,
                                                     return_result=True)

    # reconstruct masked vertices as zeros
    embedding_recort = np.zeros((len(all_voxel), embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:, e] = recort(len(all_voxel),
                                        embedding_results[:, e], brain, 0)

    return embedding_recort, embedding_dict


'''
----
RUN
----
'''

print('correlation')
ts_files = []
for sub in subjects:
    for sess in sessions:
        rest = np.load(rest_file % (sub, 'lh', sess))
        ts_files.append(rest)

upper_corr, full_shape = avg_correlation(ts_files)

print('saving matrix')
f = h5py.File(corr_file, 'w')
f.create_dataset('upper_corr', data=upper_corr)
f.create_dataset('shape', data=full_shape)
f.close()

print('embedding')
mask_img = nb.load(mask_file)
mask = mask_img.get_data()
embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, 100)

print('saving embedding')
revolume = embedding_recort.reshape(mask.shape[0], mask.shape[1],
                                    mask.shape[2], 100)
nb.Nifti1Image(revolume, mask_img.affine,
               mask_img.header).to_filename(embed_img)
np.save(embed_file, embedding_recort)
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()
