import nighres

lamina=nighres.laminar.volumetric_layering('/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/inner_p2l-surf.nii.gz', '/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/cortex_mask_tight_lvl.nii.gz', n_layers=5, save_data=True, overwrite=True, output_dir='/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/')
