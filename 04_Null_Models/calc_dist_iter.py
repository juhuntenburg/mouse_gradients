import numpy as np
import gdist
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-idx", dest="idx",required=True)
    args = parser.parse_args()

    idx = int(args.idx)

    data_dir = '/home/julia/data/gradients/'
    cortex = np.load(data_dir+'results/null_models/surface/cortex_mask.npy')
    points = np.load(data_dir+'results/null_models/surface/points.npy')
    faces = np.load(data_dir+'results/null_models/surface/faces.npy')

    dist = gdist.compute_gdist(np.array(points, dtype=np.float64),
                               np.array(faces, dtype=np.int32),
                               source_indices=np.array([cortex[idx]], dtype=np.int32),
                               target_indices=np.array(cortex[idx+1:], dtype=np.int32))

    np.save(data_dir+'results/null_models/surface/iter/idx_{}.npy'.format(idx), dist)
