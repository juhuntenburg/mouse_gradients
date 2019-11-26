import os
import pickle
from glob import glob
import numpy as np
import nibabel as nb
from nighres import io
from functions import profile_sampling


# Load previously created profiles
with open('/home/julia/data/gradients/results/profiles/profiles.pkl', 'rb') as pf:
    profiles=pickle.load(pf)

# Loop over images
func = glob("/home/julia/data/gradients/results/orig_allen/*.nii.gz")

for f in func:
    print(f)
    img = nb.load(f)
    data = img.get_data()

    data_mesh = np.zeros((len(profiles), data.shape[3]))
    for t in range(data.shape[3]):
        data_mesh[:,t] = np.squeeze(profile_sampling(data[:,:,:,t], profiles))

    # save data with nans, i.e. masked
    mf = '/home/julia/data/gradients/results/orig_mesh/%s.npy' % os.path.basename(f).split(".")[0]
    np.save(mf, data_mesh)
