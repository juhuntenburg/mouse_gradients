import os
import gc
from glob import glob
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb
import hcp_corr
from nilearn.input_data import NiftiMasker

def jo2allen_vol(data):
    data = np.swapaxes(np.swapaxes(data, 0,1), 1,2)
    data = np.flip(np.flip(data,1),0)
    return data

def avg_correlation(ts_files, thr=None):
    '''
    Calculates average connectivity matrix using hcp_corr package for memory
    optimization: https://github.com/NeuroanatomyAndConnectivity/hcp_corr
    '''
    # make empty avg corr matrix
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
        rest = np.load(rest).T

        # calculate correlations matrix
        print('...corrcoef')
        corr = hcp_corr.corrcoef_upper(rest)
        del rest

        # threshold / transform
        print('...transform')
        avg_corr += ne.evaluate('arctanh(corr)')
        del corr
        count += 1

    # divide by number of sessions included
    print('...divide')
    avg_corr /= count

    # transform back if necessary
    print('...back transform')
    avg_corr = np.nan_to_num(ne.evaluate('tanh(avg_corr)'))

    return avg_corr, full_shape


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
