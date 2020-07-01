import gc
import numpy as np
import h5py
from sklearn.decomposition import PCA

data_dir = '/home/julia/data/gradients/'
corr_file = data_dir+'results/embedding/corr.hdf5'
pc_file = data_dir+'results/pca/pca_components.npy'
var_ratio_file = data_dir+'results/pca/explained_variance_ratio.npy'
loading_file = data_dir+'results/pca/loading_matrix.npy'

print('Loading')
f = h5py.File(corr_file, 'r')
upper_corr = np.asarray(f['upper_corr'])
full_shape = tuple(f['shape'])
f.close()

print('Full Matrix')
full_corr = np.zeros(tuple(full_shape))
full_corr[np.triu_indices_from(full_corr, k=1)] = np.nan_to_num(upper_corr)
del upper_corr
gc.collect()
full_corr += full_corr.T
gc.collect()

print('PCA')
pca = PCA(n_components=100)
pcs = pca.fit_transform(full_corr)

print('Saving')
np.save(pc_file, pcs)
np.save(var_ratio_file, pca.explained_variance_ratio_)

loading_matrix = (pca.components_.T * np.sqrt(pca.explained_variance_))
np.save(loading_file, loading_matrix)
