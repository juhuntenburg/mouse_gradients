from __future__ import division
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb


ne.set_num_threads(ne.ncores-1)


ts_file = '/home/julia/projects/gradients/fisp_1/func_final.npy'
mask_file='/home/julia/projects/gradients/fisp_1/func_mask.nii.gz'
corr_file = '/home/julia/projects/gradients/fisp_1/corr.hdf5'
embed_file='/home/julia/projects/gradients/fisp_1/embed.hdf5'
embed_dict_file='/home/julia/projects/gradients/fisp_1/embed_dict.pkl'

calc_corr = False
save_corr = False
calc_embed = True

def recort(n_vertices, data, cortex, increase):
    '''
    Helper function to rewrite masked, embedded data to full cortex
    (copied from Daniel Margulies)
    '''
    d = np.zeros(n_vertices)
    count = 0
    for i in cortex:
        d[i] = data[count] + increase
        count = count +1
    return d

def embedding(masked_corr, mask, n_components):
    '''
    Diffusion embedding on connectivity matrix using mapaling package:
    https://github.com/satra/mapalign
    '''
    all_voxel=range(mask.shape[0])
    brain=np.delete(all_voxel, np.where(mask==0)[0])

    # run actual embedding
    print('...embed')
    K = (masked_corr + 1) / 2.
    del masked_corr
    K[np.where(np.eye(K.shape[0])==1)]=1.0
    embedding_results, embedding_dict = embed.compute_diffusion_map(K, n_components=n_components, overwrite=True)

    # reconstruct masked vertices as zeros
    embedding_recort=np.zeros((len(all_voxel),embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:,e]=recort(len(all_voxel), embedding_results[:,e], brain, 0)

    return embedding_recort, embedding_dict


print('correlation')
corrmat = np.corrcoef(np.load(ts_file).T)

print('saving matrix')
f = h5py.File(corr_file, 'w')
f.create_dataset('corr', data=corrmat)
f.close()


print('embedding')
mask_data = nb.load(mask_file).get_data()
volume_shape = mask_data.shape
mask = mask_data.flatten()
embedding_recort, embedding_dict = embedding(corrmat, mask, 100)

np.save(embed_file,embedding_recort)
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()
