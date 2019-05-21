from nipype.interfaces.afni import SkullStrip, Resample
from nipype.interfaces.ants import N4BiasFieldCorrection, Registration


anat = '/home/julia/projects/gradients/grandjean_data/4457_baseline/anatomical/NIFTI/T2SAnatomicalFeb15.nii.gz'
anat_corr = '/home/julia/projects/gradients/grandjean_data/4457_baseline/anatomical/NIFTI/T2SAnatomicalFeb15_corr.nii.gz'
anat_resamp = '/home/julia/projects/gradients/grandjean_data/4457_baseline/anatomical/NIFTI/T2SAnatomicalFeb15_resamp.nii.gz'
anat_stripped = '/home/julia/projects/gradients/grandjean_data/4457_baseline/anatomical/NIFTI/T2SAnatomicalFeb15_stripped.nii.gz'


n4 = N4BiasFieldCorrection(input_image=anat, output_image=anat_corr)
n4.run()

resamp = Resample(in_file=anat_corr, out_file=anat_resamp, voxel_size=(0.072959, 0.075000, ))
resamp.run()

skull = SkullStrip(in_file=anat_corr,  out_file=anat_stripped,
                   args='-rat -push_to_edge -orig_vol')
skull.run()
