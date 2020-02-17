#!/bin/bash

# Install required libraries on top of existing docker image
#pip install --user nipype

# run compression script
#echo "Start script"
#python -u $@ 1>&1 2>&2
#echo "Done script"

antsRegistration --float --collapse-output-transforms 1 --dimensionality 3 --initial-moving-transform [ /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz, /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain.nii.gz, 1 ] --initialize-transforms-per-stage 0 --interpolation BSpline --output [ yongsoo2allen_, /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/yongsoo2allen.nii.gz ] --transform Rigid[ 0.1 ] --metric MI[ /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz, /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain.nii.gz, 1, 32, Regular, 0.3 ] --convergence [ 2000x1000x500, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0vox --shrink-factors 4x2x1 --use-estimate-learning-rate-once 0 --use-histogram-matching 0 --transform Affine[ 0.1 ] --metric MI[ nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz, /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain.nii.gz, 1, 32, None, 0.3 ] --convergence [ 1000x500x250, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0vox --shrink-factors 4x2x1 --use-estimate-learning-rate-once 0 --use-histogram-matching 0 --transform SyN[ 0.1, 0.0, 3.0 ] --metric MI[ /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz, /nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain.nii.gz, 1, 4, None, 0.3 ] --convergence [ 100x50x20, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0vox --shrink-factors 4x2x1 --use-estimate-learning-rate-once 0 --use-histogram-matching 0 --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1
