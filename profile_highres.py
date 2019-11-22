import nighres

levelsets = "gradients/three_levels.nii.gz"
gradient1 = "gradients/gradient_0000_highres.nii.gz"

nighres.laminar.profile_sampling(levelsets, gradient1,
                                 save_data=True,overwrite=True)
