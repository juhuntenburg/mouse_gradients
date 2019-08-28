from nilearn.input_data import NiftiMasker
from glob import glob
import numpy as np
import os

mask = "/home/julia/data/gradients/atlas/cortex/cortex_mask_25um.nii.gz"
func = glob("/home/julia/data/gradients/orig/v2/*MEDISO*EPI*.nii.gz")

masker = NiftiMasker(mask_img=mask, standardize=True)

for f in func:
    func_compressed = masker.fit_transform(f)
    np.save('/home/julia/data/gradients/compressed/v2/%s.npy'
            % os.path.basename(f).split(".")[0], func_compressed)

    print(f)
