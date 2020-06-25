import numpy as np
import nibabel as nb
from scipy.spatial.distance import pdist
from brainsmash.mapgen.base import Base
from brainsmash.mapgen.eval import base_fit

data_dir ='/nfs/tank/shemesh/users/julia.huntenburg/revisions/brainsmash/volume/'
cortex_mask = data_dir+'cortex_mask_tight_200um.nii.gz'
gradients = data_dir+'embed.npy'

dist_file = data_dir+'cortex_euclidean.txt'
surrogate_file = data_dir+'surrogates.npy'
emp_var_file = data_dir+'emp_var.npy'
surr_var_file = data_dir+'surr_var.npy'

# Load cortex mask
cortex = np.asanyarray(nb.load(cortex_mask).dataobj)

# Make an mx3 array where each of the m entries refers to the coordinate of one voxel included in the cortex mask
coords = np.asarray(np.where(cortex==1)).T

# Calculate Euclidean distance matrix (upper triangle)
dist_triu = pdist(coords, metric='euclidean')

# Construct full matrix
dist = np.zeros(shape=(coords.shape[0], coords.shape[0]))
dist[np.triu_indices(coords.shape[0], k=1)] = dist_triu
dist += dist.T

# Write to disk
np.savetxt(dist_file, dist)

# Create base instance
g = np.load(gradients)[:,0]
base = Base(x=g, D=dist)

# Create surrogates
surrogates = base(n=1000)
np.save(surrogate_file, surrogates)

v = base.compute_variogram(g)
emp_var, u0 = base.smooth_variogram(v, return_h=True)

surr_var = np.empty((1000, generator.nh))
for i in range(1000):
    v_null = base.compute_variogram(surrogates[i])
    surr_var[i] = base.smooth_variogram(v_null, return_h=False)

np.save(emp_var_file, emp_var)
np.save(surr_var_file, surr_var)
