#!/bin/bash

ref_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template_200.nii.gz
transform=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/yongsoo2allen_Composite.h5

#input_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Parvalbumin-Vox.nii.gz
#input_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Somatostatin-Vox.nii.gz
input_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/VIP-Vox.nii.gz

#output_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/pv2allen.nii.gz
#output_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/sst2allen.nii.gz
output_img=/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/vip2allen.nii.gz


antsApplyTransforms -d 3 -i $input_img -r $ref_img -o $output_img \
--interpolation BSpline --transform $transform
