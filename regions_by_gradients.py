import numpy as np
import nibabel as nb
import pandas as pd
import pickle

# Load images and structure tree
annot = np.array(nb.load('/home/julia/data/gradients/atlas/allen_api/regions/annot_finest_200um.nii.gz').get_data(), dtype='float64')
gradient = nb.load('/home/julia/data/gradients/results/embedding_vol/embed.nii.gz').get_data()
mask = nb.load('/home/julia/data/gradients/atlas/allen_api/cortex_mask_tight_200um.nii.gz').get_data()
df = pd.read_csv('/home/julia/data/gradients/results/regions/finest_regions.csv')
df = df.drop(columns=["Unnamed: 0"])

# Caculate first the region means for each gradient
for g in range(6):

    g_masked = gradient[:,:,:,g]
    g_masked[mask==0] = np.nan

    region_means = []
    for region in df['id']:
        region_means.append(np.nanmean(g_masked[annot==region]))

    df['gradient %i' %g] = region_means

df.to_csv('/home/julia/data/gradients/results/regions/finest_regions.csv')
