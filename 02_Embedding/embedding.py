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
from embedding_functions import avg_correlation, embedding, jo2allen_vol

data_dir = '/home/julia/data/gradients/'
mask = data_dir+'allen_atlas/cortex_mask_tight_200um.nii.gz'

# Output data
corr_file = data_dir+'results/embedding/corr.hdf5'
embed_file = data_dir+'results/embeddig/embed.npy'
embed_img = data_dir+'results/embedding/embed.nii.gz'
embed_dict_file = data_dir+'results/embedding/embed_dict.pkl'

# Bring data into Allen space
img = nb.load(data_dir+'orig/sub-jgrAesMEDISOc11L_ses-1_task-rest_acq-EPI_run-1_bold.nii.gz')
data = img.get_data()
shape_allen = jo2allen_vol(data[:,:,:,0]).shape
num_vol = data.shape[3]
aff = np.eye(4)*0.2
aff[3,3]=1
hdr = nb.Nifti1Header()
hdr['dim']=np.array([4, shape_allen[0], shape_allen[1], shape_allen[2], num_vol,
                     1, 1, 1], dtype='int16')
hdr['pixdim']=img.header['pixdim']

func = glob(data_dir+'orig/*MEDISO*EPI*.nii.gz')
for f in func:
    data_allen = np.zeros((shape_allen[0], shape_allen[1], shape_allen[2], num_vol))
    data_jo = nb.load(f).get_data()

    for vol in range(num_vol):
        data_allen[:,:,:,vol] = jo2allen_vol(data_jo[:,:,:,vol])

    nb.save(nb.Nifti1Image(data_allen, aff, hdr),
            data_dir+'orig_allen/%s_allen.nii.gz' % os.path.basename(f).split(".")[0])

# Mask, smooth and compress the data
masker = NiftiMasker(mask_img=mask, standardize=True, smoothing_fwhm=0.45)

func_allen = glob(data_dir+'orig_allen/*MEDISO*EPI*.nii.gz')
for f in func_allen:
    func_compressed = masker.fit_transform(f)
    np.save(data_dir+'orig_allen/%s.npy'
            % os.path.basename(f).split(".")[0], func_compressed)
    print(f)

# Run correlation and embedding
ne.set_num_threads(ne.ncores-1)
ts_files = glob(data_dir+'orig_allen/*.npy')

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
fake_compress = masker.fit_transform(func[0])
revolume = masker.inverse_transform(embedding_result.T)
revolume.to_filename(embed_img)
