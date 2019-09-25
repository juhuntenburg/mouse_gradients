from nilearn.input_data import NiftiMasker
from glob import glob
import numpy as np
import os

mask = "/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/embedding/cortex_mask_25um.nii.gz"
func = glob("/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/embedding/orig/*MEDISO*EPI*.nii.gz")

masker = NiftiMasker(mask_img=mask, standardize=True, smoothing_fwhm=0.45)

for f in func:
    func_compressed = masker.fit_transform(f)
    np.save('/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/embedding/compressed_045/%s.npy'
            % os.path.basename(f).split(".")[0], func_compressed)

    print(f)
