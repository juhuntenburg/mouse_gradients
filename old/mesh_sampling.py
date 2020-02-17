import nighres

meshing=nighres.surface.surface_mesh_mapping("/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/vol0005.nii.gz", "/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/brain_mesh.vtk", mapping_method="closest_point", save_data=True, overwrite=True, output_dir='/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/layers/')
