from glob import glob
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb
import hcp_corr
import gc
from nighres import io


ne.set_num_threads(ne.ncores-1)

# inputs
ts_files = glob('/home/julia/data/gradients/results/orig_mesh/*.npy')
brain_mesh = '/home/julia/data/gradients/atlas/allen_api/brain_mesh.vtk'

# outputs
corr_file = '/home/julia/data/gradients/results/embedding/corr.hdf5'
embed_file = '/home/julia/data/gradients/results/embedding/embed.npy'
embed_mesh = '/home/julia/data/gradients/results/embedding/embed.vtk'
embed_dict_file = '/home/julia/data/gradients/results/embedding/embed_dict.pkl'



def avg_correlation(ts_files, mask, thr=None):
    '''
    Calculates average connectivity matrix using hcp_corr package for memory
    optimization: https://github.com/NeuroanatomyAndConnectivity/hcp_corr
    '''

    # make empty avg corr matrix
    data0 = np.load(ts_files[0])
    get_size = data0[mask].shape[0]
    del data0

    full_shape = (get_size, get_size)
    if np.mod((get_size**2-get_size), 2) == 0.0:
        avg_corr = np.zeros(int((get_size**2-get_size)/2))
    else:
        print('size calculation no zero mod')

    count = 0
    for rest in ts_files:

        # load time series
        rest = np.load(rest)

        # mask time series
        rest = rest[mask]

        # calculate correlations matrix
        print('...corrcoef')
        corr = hcp_corr.corrcoef_upper(rest)
        del rest

        # r-to-z transform and add to avg
        print('...transform')
        avg_corr += ne.evaluate('arctanh(corr)')

        count += 1

    # divide by number of sessions included
    print('...divide')
    avg_corr /= count

    # transform back
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


'''
----
RUN
----
'''
print('mask mask')
data0 = np.load(ts_files[0])
mask = np.where(np.isnan(data0[:,0])==False)

print('correlation')
upper_corr, full_shape = avg_correlation(ts_files, mask)

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
n = 100
embedding_result, embedding_dict = embedding(upper_corr, full_shape, n)

print('saving embedding')
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()
np.save(embed_file, embedding_result)

print('unmask')
mesh = io.load_mesh_geometry(brain_mesh)
full_data = np.zeros((mesh['points'].shape[0],n))
full_data[mask] = embedding_result
mesh['data'] = full_data
io.save_mesh(embed_mesh, mesh)
