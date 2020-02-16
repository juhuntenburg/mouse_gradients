from nipype.interfaces.ants import Registration

allen_img = "/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/template.nii.gz"
yongsoo_img = "/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/Reference-brain.nii.gz"
warped_img = "/nfs/tank/shemesh/users/julia.huntenburg/rodent_gradients/ants/yongsoo2allen.nii.gz"

reg_struct = Registration(fixed_image = allen_img,
                          moving_image = yongsoo_img,
                          output_warped_image = warped_img,
                          output_transform_prefix = "yongsoo2allen_",
                          dimension = 3,
                          transforms = ['Rigid', 'Affine', 'SyN'],
                          metric = ['MI', 'MI', 'MI'],
                          transform_parameters = [(0.1,), (0.1,), (0.1, 0, 3.0)],
                          metric_weight = [1, 1, 1],
                          radius_or_number_of_bins = [32, 32, 4],
                          sampling_percentage = [0.3, 0.3, 0.3],
                          sampling_strategy = ['Regular', None, None],
                          convergence_threshold = [1.e-6, 1.e-6, 1.e-6],
                          convergence_window_size = [10, 10, 10],
                          smoothing_sigmas = [[3,2,1], [3,2,1], [3,2,1]],
                          sigma_units = ['vox', 'vox', 'vox'],
                          shrink_factors = [[4, 2, 1], [4, 2, 1], [4, 2, 1]],
                          use_estimate_learning_rate_once = [False, False, False],
                          use_histogram_matching = [False, False, False],
                          number_of_iterations = [[2000, 1000, 500], [1000, 500, 250], [100, 50, 20]],
                          write_composite_transform = True,
                          collapse_output_transforms = True,
                          winsorize_lower_quantile = 0.005,
                          winsorize_upper_quantile = 0.995,
                          args = '--float',
                          num_threads = 16,
                          interpolation = 'BSpline',
                          initial_moving_transform_com = True,
                         )
reg_struct.run()
