from glob import glob
from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.nipy as nipy
import nipype.interfaces.ants as ants
import nipype.interfaces.afni as afni
import nipype.interfaces.freesurfer as fs
import nipype.algorithms.rapidart as ra
from functions import strip_rois_func, motion_regressors, median, selectindex, nilearn_denoise, weighted_avg, pca_denoising

vol_to_remove = 50
motion_norm = 0.3
z_thr = 3
tr = 1.84
bandpass = [0.1, 0.01]  # rather 0.3 but too high for our temporal resolution

# directories
working_dir = '/home/julia/projects/gradients/working_dir/'
data_dir= '/home/julia/projects/gradients/fisp_1/'
out_dir = '/home/julia/projects/gradients/fisp_1/'

# main workflow
preproc = Workflow(name='preproc')
preproc.base_dir = working_dir
preproc.config['execution']['crashdump_dir'] = preproc.base_dir + "/crash_files"

# select files
templates = {'rest' : 'JH20181220lcrsfmritest03_19.nii.gz',
           }
selectfiles = Node(nio.SelectFiles(templates, base_directory=data_dir),
                   name="selectfiles")


############################
# Functional preprocessing #
############################


# Remove first volumes
remove_vol = Node(util.Function(input_names=['in_file','t_min'],
                                output_names=["out_file"],
                                function=strip_rois_func),
                  name='remove_vol')
remove_vol.inputs.t_min = vol_to_remove
preproc.connect([(selectfiles, remove_vol, [('rest', 'in_file')])])

# Thermal noise removal
func_denoise = Node(util.Function(input_names=['in_file'],
                                    output_names=['denoised_data', 'sigmas',
                                                  'preserved_components'],
                                     function=pca_denoising),
                                     name='func_denoise')
preproc.connect([(remove_vol, func_denoise, [('out_file', 'in_file')])])

# motion correction
moco = Node(nipy.SpaceTimeRealigner(slice_times='asc_alt_2', tr=tr, slice_info=[2,1]),name="moco")
preproc.connect([(func_denoise, moco, [('denoised_data', 'in_file')])])
#preproc.connect([(remove_vol, moco, [('out_file', 'in_file')])])


# compute median
median = Node(util.Function(input_names=['in_files'],
                       output_names=['median_file'],
                       function=median),
              name='median')

preproc.connect([(moco, median, [('out_file', 'in_files')])])

# bias field correction
biasfield = Node(ants.N4BiasFieldCorrection(dimension=3,
                    n_iterations=[150,100,50,30], convergence_threshold=1e-11,
                    bspline_fitting_distance = 10, bspline_order = 4,
                    shrink_factor = 2, output_image='func_median.nii.gz'),name='biasfield')
preproc.connect([(median, biasfield, [('median_file', 'input_image')])])

# resample to isotropic (otherwise too few slices for afni SkullStrip)
iso = Node(fsl.FLIRT(apply_isoxfm=0.2, out_file='func_median_iso.nii.gz'),
           name='iso')

preproc.connect([(biasfield, iso, [('output_image', 'in_file'),
                                   ('output_image', 'reference')])])

# create functional mask
func_mask = Node(afni.SkullStrip(outputtype='NIFTI_GZ',
                                 args='-rat -orig_vol'),
                 name='func_mask')
preproc.connect([(iso, func_mask, [('out_file','in_file')])])

# make binary mask, fill holes and resample to original resolution
binarize = Node(fsl.maths.MathsCommand(args='-bin -dilM -ero'),
                name='binarize')
preproc.connect([(func_mask, binarize, [('out_file', 'in_file')])])

resamp = Node(afni.Resample(resample_mode='NN', out_file='func_mask.nii.gz'),
                name='resamp')
preproc.connect([(binarize, resamp, [('out_file', 'in_file')]),
                 (biasfield, resamp, [('output_image', 'master')])])


# artefact detection
artefact = Node(ra.ArtifactDetect(save_plot=True,
                                  use_norm=True,
                                  parameter_source='NiPy',
                                  mask_type='file',
                                  norm_threshold=motion_norm,
                                  zintensity_threshold=z_thr,
                                  use_differences=[True,False]),
                 name='artefact')

preproc.connect([(moco, artefact, [('out_file', 'realigned_files'),
                                   ('par_file', 'realignment_parameters')]),
                 (resamp, artefact, [('out_file', 'mask_file')]),
                 ])

# calculate motion regressors
motreg = Node(util.Function(input_names=['motion_params', 'order','derivatives'],
                            output_names=['out_files'],
                            function=motion_regressors),
                 name='motion_regressors')
motreg.inputs.order=2
motreg.inputs.derivatives=2
preproc.connect([(moco, motreg, [('par_file','motion_params')])])

# nuissance regression
regress = Node(util.Function(input_names=['in_file', 'brain_mask',
                                          'motreg_file', 'outlier_file',
                                          'bandpass', 'tr'],
                             output_names=['denoised_img', 'denoised_data',
                                           'confounds'],
                             function=nilearn_denoise), name='regress')
regress.inputs.tr = tr
regress.inputs.bandpass = bandpass

preproc.connect([(moco, regress, [('out_file', 'in_file')]),
                 (resamp, regress, [('out_file', 'brain_mask')]),
                 (motreg, regress, [(('out_files',selectindex,[0]), 'motreg_file')]),
                 (artefact, regress, [('outlier_files', 'outlier_file')])
                 ])
#
#
# ############################
# # Structural preprocessing #
# ############################
#
# # Thermal noise removal
# # struct_denoise = Node(util.Function(input_names=['in_file'],
# #                                     output_names=['denoised_data', 'sigmas',
# #                                                   'preserved_components'],
# #                                      function=pca_denoising),
# #                                      name='struct_denoise')
# # preproc.connect([(selectfiles, struct_denoise, [('brain', 'in_file')])])
# #
# # # Split structural image in individual echo times
# # img_split = Node(fsl.Split(dimension='t', output_type='NIFTI_GZ'),
# #                  name='img_split')
# # preproc.connect([(struct_denoise, img_split, [('denoised_data', 'in_file')])])
# #
# # # Bias field correction of each echo time
# # struct_bias = MapNode(ants.N4BiasFieldCorrection(dimension=3,
# #                                                  n_iterations=[100,100,100,100],
# #                                                  convergence_threshold=0.0,),
# #                                                  iterfield=['input_image'],
# #                                                  name='struct_bias')
# #
# # preproc.connect([(img_split, struct_bias, [('out_files', 'input_image')])])
# #
# # # Merge corrected files again
# # img_merge = Node(fsl.Merge(dimension='t', output_type='NIFTI_GZ', merged_file='struct_corr.nii.gz'),
# #                  name='img_merge')
# # preproc.connect([(struct_bias, img_merge, [('output_image','in_files')])])
# #
# # # Create average across all echo times
# # average = Node(fsl.MeanImage(out_file='struct_corr_avg.nii.gz'), name='struct_average')
# # preproc.connect([(img_merge, average, [('merged_file','in_file')])])
# #
# # # Skull stripping on first echo time (highest SNR)
# # skullstrip = Node(afni.SkullStrip(outputtype='NIFTI_GZ',
# #                                   args='-rat -push_to_edge -orig_vol'),
# #                   name='skullstrip')
# # preproc.connect([(struct_bias, skullstrip, [(('output_image', selectindex, [0]),
# #                                               'in_file')])])
# #
# # # Binarize mask
# # struct_mask = Node(fs.Binarize(out_type = 'nii.gz', min=0.1, binary_file='struct_mask.nii.gz'), name='struct_mask')
# # preproc.connect([(skullstrip, struct_mask, [('out_file','in_file')])])
# #
# # # Create masked, weighted image for coregistration
# # weighted_avg = Node(util.Function(input_names=['in_file', 'mask_file'],
# #                                   output_names=['out_file'],
# #                                   function=weighted_avg),
# #                                   name='weighted_avg')
# # preproc.connect([(img_merge, weighted_avg, [('merged_file', 'in_file')]),
# #                  (struct_mask, weighted_avg, [('binary_file', 'mask_file')])])
# #
#
# ################
# # Registration #
# ################
#
#
# #############
# # Save data #
# #############
#
# # Sink relevant files
func_sink = Node(nio.DataSink(parameterization=False),name='func_sink')
func_sink.inputs.base_directory = out_dir
func_sink.inputs.regexp_substitutions = [('corr_.*_roi_mp_denoised_denoised', 'func_final'),
                                      ('corr_.*_roi_mp_denoised','func_moco')]
preproc.connect([(moco, func_sink, [('out_file', '@realigned_file'),
                                    ('par_file', 'confounds.@orig_motion')]),
                 (resamp, func_sink, [('out_file', '@mask')]),
                 (biasfield, func_sink, [('output_image', '@median')]),
                 (artefact, func_sink, [('norm_files', 'confounds.@norm_motion'),
                                   ('outlier_files', 'confounds.@outlier_files'),
                                   ('intensity_files', 'confounds.@intensity_files'),
                                   ('statistic_files', 'confounds.@outlier_stats'),
                                   ('plot_files', 'confounds.@outlier_plots')]),
                 (motreg, func_sink, [('out_files', 'confounds.@motreg')]),
                 (regress, func_sink, [('denoised_img', '@denoised_img'),
                                  ('denoised_data', '@denoised_data'),
                                  ('confounds', 'confounds.@confounds')])])
#
# # struct_sink = Node(nio.DataSink(parameterization=False),name='struct_sink')
# # struct_sink.inputs.base_directory = out_dir + dataset + '/struct'
# # preproc.connect([(struct_mask, struct_sink, [('binary_file', '@mask')]),
# #                  (img_merge, struct_sink, [('merged_file', '@corrected')]),
# #                  (average, struct_sink, [('out_file', '@corrected_avg')]),
# #                  (weighted_avg, struct_sink, [('out_file', '@weighted_avg')])
# #                  ])


preproc.run() #plugin='MultiProc', plugin_args={'n_procs' : 2})
