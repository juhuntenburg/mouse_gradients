import numpy as np
import nibabel as nb
import pandas as pd
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

# Load images and structure tree
annot = np.array(nb.load('/home/julia/data/gradients/atlas/allen_api/annotation.nii.gz').get_data(), dtype='float64')
mask = nb.load('/home/julia/data/gradients/atlas/cortex/cortex_mask_25um_allen.nii.gz').get_data()

mcc = MouseConnectivityCache(manifest_file = '/home/julia/data/gradients/atlas/allen_api/mouse_connectivity_manifest.json')
structure_tree = mcc.get_structure_tree()
structure_map = structure_tree.get_name_map()

# Finding the unique structures within the cortex mask
annot[mask==0] = np.nan
unique_annot = np.unique(annot)
unique_annot = unique_annot[np.isnan(unique_annot)==0]
unique_annot = unique_annot[1:]

# Finding the parent structures (for more coarse grained regions)
parents = []
for child in unique_annot:
    parents.append(structure_tree.parent_ids([child])[0])
parents=np.array(parents)

# Load list of curated parents (see notebook)
curated_parents = list(np.load('/home/julia/data/gradients/results/regions_by_gradient/curated_parents.npy'))

# Caculate first the child and the the parent region means for each gradient
for g in range(6):
    gradient = nb.load('/home/julia/data/gradients/results/gradient000%s_highres_allen.nii.gz' %str(g)).get_data()

    region_means = []
    for region in range(unique_annot.shape[0]):
        region_means.append(np.mean(gradient[annot==unique_annot[region]]))
    region_means = np.array(region_means)

    parent_means = []
    parent_names = []
    for parent in curated_parents:
        parent_means.append(np.mean(region_means[parents==parent]))
        parent_names.append(structure_map[parent])

    df = pd.DataFrame(data=zip(curated_parents, parent_names, parent_means),
                      columns=['idx', 'name','mean'])

    df = df.sort_values(by='mean')

    df.to_csv('/home/julia/data/gradients/results/regions_by_gradient/curated_means_gradient000%s.csv' %str(g))
