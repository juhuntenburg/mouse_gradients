import nighres

levelsets = "layers/upper_lower_lvl.nii.gz"
gradient1 = "layers/gradient_0000_highres.nii.gz"

nighres.laminar.profile_sampling(levelsets, gradient1,
                                save_data=True,overwrite=True)
