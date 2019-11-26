import numpy as np
import nibabel as nb
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# Load images and structure tree
annot_img = nb.load('/home/julia/data/gradients/atlas/allen_api/annotation.nii.gz')
aff, hdr = annot_img.affine, annot_img.header
annot = np.array(annot_img.get_data(), dtype='float32')
mcc = MouseConnectivityCache(manifest_file = '/home/julia/data/gradients/atlas/allen_api/mouse_connectivity_manifest.json')
structure_tree = mcc.get_structure_tree()
structure_map = structure_tree.get_name_map()

# For each layer
for l in ['l1', 'l2_3', 'l4', 'l5', 'l6a', 'l6b']:
    # Finding the unique structures within the mask
    mask = nb.load('/home/julia/data/gradients/atlas/allen_api/%s.nii.gz' %l).get_data()
    annot = np.array(annot_img.get_data(), dtype='float32')
    annot[mask==0] = np.nan
    unique_annot = np.unique(annot)
    unique_annot = unique_annot[np.isnan(unique_annot)==0]
    #unique_annot = unique_annot[1:] ################## double check

# Caculate first the region means
    gradient = nb.load('/home/julia/data/gradients/results/gradient0000_highres_allen.nii.gz').get_data()

    region_means = []
    region_names = []
    for region in range(unique_annot.shape[0]):
        try:
            region_names.append(structure_map[unique_annot[region]])
            region_means.append(np.mean(gradient[annot==unique_annot[region]]))
        except:
            pass
    region_means = np.array(region_means)


    df = pd.DataFrame(data=zip(unique_annot, region_names, region_means),
                      columns=['idx', 'name', 'mean'])
    df = df.sort_values(by='mean')

    df.to_csv('/home/julia/data/gradients/results/regions_by_gradient/region_means_gradient0000_%s.csv' %l)
