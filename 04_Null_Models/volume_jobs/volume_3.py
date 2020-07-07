import pickle
import numpy as np
from joblib import Parallel, delayed
from brainsmash.mapgen.sampled import Sampled


def compute_surrogates(n_gradient, n_perms):

    data_dir = '/nfs/tank/shemesh/users/julia.huntenburg/revisions/null_models/volume/'
    gradients = np.load(data_dir+'embed.npy')[:,:6]
    distmat = data_dir+'distmat.npy'
    index = data_dir+'index.npy'
    dict_file = data_dir+'vol{}_n{}.pkl'

    nsurr = 1000
    surr_dict ={}

    # Construct sampled class
    sampled = Sampled(gradients[:,n_gradient], distmat, index,
                         deltas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                         ns=1000,
                         knn=5000)

    # Compute and save surrogates
    surrogates = sampled(n=nsurr)
    surr_dict['maps'] = surrogates

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

    surr_dict['emp_var']=emp_var_samples
    surr_dict['u0']=u0_samples
    surr_dict['surr_var']=surr_var

    pkl_out = open(dict_file.format(n_gradient, n_perms), 'wb')
    pickle.dump(surr_dict, pkl_out)
    pkl_out.close()


Parallel(n_jobs=10)(delayed(compute_surrogates)(3,i) for i in range(10))
