from nilearn.input_data import NiftiMasker
from glob import glob
import numpy as np
import os

mask = "/home/julia/projects/gradients/data_jo/atlas/isocortex.nii.gz"
func = glob("/home/julia/projects/gradients/data_jo/orig/*MEDISO*.nii.gz")

masker = NiftiMasker(mask_img=mask, standardize=True)

for f in func:
    func_compressed = masker.fit_transform(f)
    np.save('/home/julia/projects/gradients/data_jo/compressed/%s.npy'
            % os.path.basename(f).split(".")[0], func_compressed)

    print(f)
