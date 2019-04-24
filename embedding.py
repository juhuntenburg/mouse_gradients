from __future__ import division
import numpy as np
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import nibabel as nb
import hcp_corr


ne.set_num_threads(ne.ncores-1)


ts_file = '/home/julia/projects/gradients/fisp_1/func_final.npy'
mask_file='/home/julia/projects/gradients/fisp_1/func_mask.nii.gz'
corr_file = '/home/julia/projects/gradients/fisp_1/corr.hdf5'
embed_file='/home/julia/projects/gradients/fisp_1/embed.npy'
embed_img='/home/julia/projects/gradients/fisp_1/embed.nii.gz'
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

def embedding(upper_corr, full_shape, mask, n_components):
    '''
    Diffusion embedding on connectivity matrix using mapaling package:
    https://github.com/satra/mapalign
    '''
    # reconstruct full matrix
    print '...full matrix'
    full_corr = np.zeros(tuple(full_shape))
    full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
    full_corr += full_corr.T

    mask_flat = mask.flatten()
    all_voxel = range(mask_flat.shape[0])
    brain = np.delete(all_voxel, np.where(mask_flat==0)[0])


    # run actual embedding
    print '...embed'
    K = (full_corr + 1) / 2.
    del full_corr
    K[np.where(np.eye(K.shape[0])==1)]=1.0
    #v = np.sqrt(np.sum(K, axis=1))
    #A = K/(v[:, None] * v[None, :])
    #del K, v
    #A = np.squeeze(A * [A > 0])
    #embedding_results = runEmbed(A, n_components_embedding)
    embedding_results, embedding_dict = embed.compute_diffusion_map(K, n_components=n_components, overwrite=True, return_result=True)

    # reconstruct masked vertices as zeros
    embedding_recort=np.zeros((len(all_voxel),embedding_results.shape[1]))
    for e in range(embedding_results.shape[1]):
        embedding_recort[:,e]=recort(len(all_voxel), embedding_results[:,e], brain, 0)

    return embedding_recort, embedding_dict


print('correlation')
ts = np.load(ts_file).T
get_size = ts.shape[0]
full_shape = (get_size, get_size)
upper_corr = hcp_corr.corrcoef_upper(ts)

print('saving matrix')
f = h5py.File(corr_file, 'w')
f.create_dataset('upper_corr', data=upper_corr)
f.close()

print('embedding')
mask_img = nb.load(mask_file)
mask = mask_img.get_data()
embedding_recort, embedding_dict = embedding(upper_corr, full_shape, mask, 100)

print('saving embedding')
revolume = embedding_recort.reshape(mask.shape[0], mask.shape[1], mask.shape[2], 100)
nb.Nifti1Image(revolume, mask_img.affine, mask_img.header).to_filename(embed_img)
np.save(embed_file,embedding_recort)
pkl_out = open(embed_dict_file, 'wb')
pickle.dump(embedding_dict, pkl_out)
pkl_out.close()
