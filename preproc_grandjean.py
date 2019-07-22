import nipype.interfaces.ants as ants
import nipype.interfaces.afni as afni
from functions import median


ts_file = '/home/julia/projects/gradients/grandjean_data/4457_baseline/rsfMRI/NIFTI/altplusz2/meica_medn.nii.gz'

median_file = median(ts_file)

# biasfield = ants.N4BiasFieldCorrection(input_image=median_file, dimension=3,
#                                        n_iterations=[150, 100, 50, 30],
#                                        convergence_threshold=1e-11,
#                                        bspline_fitting_distance=10,
#                                        bspline_order=4, shrink_factor=2)
# res = biasfield.run()

func_mask = afni.Automask(in_file=median_file,  # res.outputs.output_image,
                          args='-peels 4', outputtype='NIFTI_GZ',
                          out_file='func_mask.nii.gz')

mask = func_mask.run()
