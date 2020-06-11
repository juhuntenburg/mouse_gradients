import numpy as np
import nibabel as nb
from glob import glob
import os
import gc
import h5py
import pickle
from mapalign import embed
import numexpr as ne
import hcp_corr
from nilearn.input_data import NiftiMasker
from embedding_functions import avg_correlation, embedding, jo2allen_vol


data_dir = '/home/julia/data/gradients/'
datasets = ['csd', 'ad2', 'ad3']

for d in datasets:
    # Transfrom from Jo space to Allen space
    aff = nb.load(atlas/allen_api/template_200.nii.gz').affine
    hdr = nb.load(data_dir+'allen_atlas/template_200um.nii.gz').header

    orig = glob(data_dir+'repro/%s/%s*.nii.gz'%(d, d.upper()))
    for img in orig:
        data = nb.load(img).get_data()
        nb.save(nb.Nifti1Image(jo2allen_vol(data), aff, hdr),
                data_dir+'repro/%s_allen/%s_allen.nii.gz' % (d, os.path.basename(img).split(".")[0]))

    # Input data
    mask = data_dir+'allen_atlas/cortex_mask_tight_200um.nii.gz'
    func = glob(data_dir+'/repro/%s_allen/*.nii.gz' %d)

    # Output data
    corr_file = data_dir+'results/repro/%s_corr.hdf5'
    embed_file = data_dir+'results/repro/%s_embed.npy'
    embed_img = data_dir+'results/repro/%s_embed.nii.gz'
    embed_dict_file = data_dir+'results/repro/%s_embed_dict.pkl'


    # Mask, smooth and compress the data
    masker = NiftiMasker(mask_img=mask, standardize=True, smoothing_fwhm=0.45)

    for f in func:
        func_compressed = masker.fit_transform(f)
        np.save(data_dir+'repro/%s_allen/%s.npy'
                % (d, os.path.basename(f).split(".")[0], func_compressed))
        print(f)

    # Run correlation and embedding
    ne.set_num_threads(ne.ncores-1)
    ts_files = glob(data_dir+'repro/%s_allen/*.npy' %d)

    print('correlation')
    upper_corr, full_shape = avg_correlation(ts_files)

    print('saving matrix')
    f = h5py.File(corr_file%d, 'w')
    f.create_dataset('upper_corr', data=upper_corr)
    f.create_dataset('shape', data=full_shape)
    f.close()

    print('embedding')
    embedding_result, embedding_dict = embedding(upper_corr, full_shape, 100)

    print('saving embedding')
    pkl_out = open(embed_dict_file%d, 'wb')
    pickle.dump(embedding_dict, pkl_out)
    pkl_out.close()
    np.save(embed_file%d, embedding_result)

    print('revolume')
    masker = NiftiMasker(mask_img=mask, standardize=True)
    fake_compress = masker.fit_transform(func[0])
    revolume = masker.inverse_transform(embedding_result.T)
    revolume.to_filename(embed_img%d)
