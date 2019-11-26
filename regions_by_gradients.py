import numpy as np
import nibabel as nb
import pandas as pd
import pickle

# Load images and structure tree
regions = np.array(nb.load('/home/julia/data/gradients/atlas/allen_api/regions/isocortex_annot_finer_200um.nii.gz').get_data(), dtype='float64')
gradient = nb.load('/home/julia/data/gradients/results/embedding_vol/embed.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight_200um.nii.gz').get_data()
with open('/home/julia/data/gradients/atlas/allen_api/regions/isocortex_annot_finer.pkl', 'rb') as f:
    regions_dict = pickle.load(f)

# Caculate first the region means for each gradient
for g in range(6):

    g_masked = gradient[:,:,:,g]
    g_masked[mask==0] = np.nan

    region_means = []
    for region in regions_dict.values():
        region_means.append(np.nanmean(g_masked[regions==region]))
    region_means = np.array(region_means)

    df = pd.DataFrame(data=zip(regions_dict.keys(), regions_dict.values(), region_means),
                      columns=['region', 'idx','mean'])

    df = df.sort_values(by='mean')

    df.to_csv('/home/julia/data/gradients/results/regions/finer_regions_by_gradients_%s.csv' %str(g))
