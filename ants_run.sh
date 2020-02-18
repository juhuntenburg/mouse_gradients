#!/bin/bash

# Install required libraries on top of existing docker image
#pip install --user nipype

# run compression script
#echo "Start script"
#python -u $@ 1>&1 2>&2
#echo "Done script"
allen_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz
yongsoo_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain_masked_clipped.nii.gz
yongsoo_img_warped=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/yongsoo2allen.nii.gz

antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 \
--initial-moving-transform [ $allen_img, $yongsoo_img, 1 ] \
--initialize-transforms-per-stage 0 \
--interpolation BSpline \
--output [ yongsoo2allen_, $yongsoo_img_warped ] \
--use-estimate-learning-rate-once 0 \
--use-histogram-matching 0 \
--winsorize-image-intensities [ 0.005, 0.995 ]  \
--write-composite-transform 1 \
--transform Rigid[ 0.1 ] \
--metric MI[ $allen_img, $yongsoo_img, 1, 32, Regular, 0.25 ] \
--convergence [ 1000x500x250x100, 1e-06, 10 ] \
--smoothing-sigmas 3x2x1x0vox \
--shrink-factors 8x4x2x1 \
--transform Affine[ 0.1 ] \
--metric MI[ $allen_img, $yongsoo_img, 1, 32, Regular, 0.25 ] \
--convergence [ 1000x500x250x100,1e-6,10 ] \
--smoothing-sigmas 3x2x1x0vox \
--shrink-factors 8x4x2x1 \
--transform SyN[ 0.1, 3, 0 ] \
--metric CC[ $allen_img, $yongsoo_img, 1, 4 ] \
--convergence [ 100x70x50x20,1e-6,10 ] \
--smoothing-sigmas 3x2x1x0vox \
--shrink-factors 8x4x2x1 \
