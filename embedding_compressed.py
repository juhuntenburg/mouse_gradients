from __future__ import division
from glob import glob
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb
import hcp_corr
import gc
from nilearn.input_data import NiftiMasker


ne.set_num_threads(ne.ncores-1)

ts_vol = "/data/julia/data_jo/in/orig/sub-jgrAesMEDISOc21R1L_ses-2_task-rest_acq-EPI_run-2_bold.nii.gz"
ts_files = glob('/data/julia/data_jo/in/compressed/*.npy')
corr_file = '/data/julia/data_jo/out/compressed/corr.hdf5'
embed_file = '/data/julia/data_jo/out/compressed/embed.npy'
embed_img = '/data/julia/data_jo/out/compressed/embed.nii.gz'
embed_dict_file = '/data/julia/data_jo/out/compressed/embed_dict.pkl'
mask = "/data/julia/data_jo/in/cortex_mask.nii.gz"


def avg_correlation(ts_files, thr=None):
    '''
    Calculates average connectivity matrix using hcp_corr package for memory
    optimization: https://github.com/NeuroanatomyAndConnectivity/hcp_corr
    '''
    # make empty avg corr matrix
    if type(ts_files[0]) == str:
        img0 = np.load(ts_files[0])
        get_size = img0.shape[1]
        del img0

    full_shape = (get_size, get_size)
    if np.mod((get_size**2-get_size), 2) == 0.0:
        avg_corr = np.zeros(int((get_size**2-get_size)/2))
    else:
        print('size calculation no zero mod')

    count = 0
    for rest in ts_files:
        # load time series
        if type(rest) == str:
            rest = np.load(rest).T
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


def embedding(upper_corr, full_shape, n_components):
    '''
    Diffusion embedding on connectivity matrix using mapaling package:
    https://github.com/satra/mapalign
    '''
    # reconstruct full matrix
    print('...full matrix')
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    del upper_corr
    gc.collect()
    full_corr += full_corr.T
    gc.collect()

    # run actual embedding
    print('...embed')
    K = (full_corr + 1) / 2.
    del full_corr
    gc.collect()
    K[np.where(np.eye(K.shape[0]) == 1)] = 1.0

    embedding_results, \
        embedding_dict = embed.compute_diffusion_map(K,
                                                     n_components=n_components,
                                                     overwrite=True,
                                                     return_result=True)

    return embedding_results, embedding_dict


'''
----
RUN
----
'''

print('correlation')
upper_corr, full_shape = avg_correlation(ts_files)

print('saving matrix')
f = h5py.File(corr_file, 'w')
f.create_dataset('upper_corr', data=upper_corr)
f.create_dataset('shape', data=full_shape)
f.close()

# print('loading matrix')
# f = h5py.File(corr_file, 'r')
# upper_corr = np.asarray(f['upper_corr'])
# full_shape = tuple(f['shape'])
# f.close()

print('embedding')
embedding_result, embedding_dict = embedding(upper_corr, full_shape, 100)

print('saving embedding')
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()
np.save(embed_file, embedding_result)

print('revolume')
masker = NiftiMasker(mask_img=mask, standardize=True)
fake_compress = masker.fit_transform(ts_vol)
revolume = masker.inverse_transform(embedding_result.T)
revolume.to_filename(embed_img)
