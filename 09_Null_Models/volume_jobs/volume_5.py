import numpy as np
from brainsmash.mapgen.sampled import Sampled

data_dir = '/nfs/tank/shemesh/users/julia.huntenburg/revisions/null_models/volume/'
gradients = np.load(data_dir+'embed.npy')[:,:6]
distmat = data_dir+'distmat.npy'
index = data_dir+'index.npy'
surr_file = data_dir+'vol5_surrogates.npy'
emp_var_file = data_dir+'vol5_emp_var.npy'
surr_var_file = data_dir+'vol5_surr_var.npy'
u0_file = data_dir+'vol5_u0.npy'

nsurr = 100000

# Construct sampled class
sampled = Sampled(gradients[:,5], distmat, index,
                     deltas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     ns=1000,
                     knn=5000)


# Compute and save surrogates
surrogates = sampled(n=nsurr)
np.save(surr_file, surrogates)

# Compute and saving variograms
surr_var = np.empty((nsurr, sampled.nh))
emp_var_samples = np.empty((nsurr, sampled.nh))
u0_samples = np.empty((nsurr, sampled.nh))
for i in range(nsurr):
    idx = sampled.sample()  # Randomly sample a subset of brain areas
    v = sampled.compute_variogram(sampled.x, idx)
    u = sampled.D[idx, :]
    umax = np.percentile(u, sampled.pv)
    uidx = np.where(u < umax)
    emp_var_i, u0i = sampled.smooth_variogram(
        u=u[uidx], v=v[uidx], return_h=True)
    emp_var_samples[i], u0_samples[i] = emp_var_i, u0i
    # Surrogate
    v_null = sampled.compute_variogram(surrogates[i], idx)
    surr_var[i] = sampled.smooth_variogram(
        u=u[uidx], v=v_null[uidx], return_h=False)

np.save(emp_var_file, emp_var_samples)
np.save(u0_file, u0_samples)
np.save(surr_var_file, surr_var)
