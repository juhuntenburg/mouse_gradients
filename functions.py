def fix_hdr(data_file, header_file):
    '''
    Overwrites the header of data_file with the header of header_file
    USE WITH CAUTION
    '''

    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename

    data=nb.load(data_file).get_data()
    hdr=nb.load(header_file).get_header()
    affine=nb.load(header_file).get_affine()

    new_file=nb.Nifti1Image(data, affine, hdr)
    _, base, _ = split_filename(data_file)
    nb.save(new_file, base + "_fixed.nii.gz")
    return os.path.abspath(base + "_fixed.nii.gz")


def nilearn_denoise(in_file, brain_mask, motreg_file, outlier_file, bandpass, tr):
    """Clean time series using Nilearn high_variance_confounds to extract
    CompCor regressors and NiftiMasker for regression of all nuissance regressors,
    detrending, normalziation and bandpass filtering.
    """
    import numpy as np
    import nibabel as nb
    import os
    from nilearn.image import high_variance_confounds
    from nilearn.input_data import NiftiMasker
    from nipype.utils.filemanip import split_filename

    # reload niftis to round affines so that nilearn doesn't complain
    #csf_nii=nb.Nifti1Image(nb.load(csf_mask).get_data(), np.around(nb.load(csf_mask).get_affine(), 2), nb.load(csf_mask).get_header())
    #time_nii=nb.Nifti1Image(nb.load(in_file).get_data(),np.around(nb.load(in_file).get_affine(), 2), nb.load(in_file).get_header())

    # infer shape of confound array
    confound_len = nb.load(in_file).get_data().shape[3]

    # create outlier regressors
    outlier_regressor = np.empty((confound_len,1))
    try:
        outlier_val = np.genfromtxt(outlier_file)
    except IOError:
        outlier_val = np.empty((0))
    for index in np.atleast_1d(outlier_val):
        outlier_vector = np.zeros((confound_len, 1))
        outlier_vector[int(index)] = 1
        outlier_regressor = np.hstack((outlier_regressor, outlier_vector))

    outlier_regressor = outlier_regressor[:,1::]

    # load motion regressors
    motion_regressor=np.genfromtxt(motreg_file)

    # extract high variance confounds in wm/csf masks from motion corrected data
    #csf_regressor=high_variance_confounds(time_nii, mask_img=csf_nii, detrend=True)
    #global_regressor=high_variance_confounds(time_nii, mask_img=brain_nii, detrend=True)

    # create Nifti Masker for denoising
    denoiser=NiftiMasker(mask_img=brain_mask, standardize=True, detrend=True, high_pass=bandpass[1], low_pass=bandpass[0], t_r=tr)

    # denoise and return denoise data to img
    confounds=np.hstack((outlier_regressor, motion_regressor)) # csf regressor
    denoised_data=denoiser.fit_transform(in_file, confounds=confounds)
    denoised_img=denoiser.inverse_transform(denoised_data)

    # save
    _, base, _ = split_filename(in_file)
    img_fname = base + '_denoised.nii.gz'
    nb.save(denoised_img, img_fname)

    data_fname = base + '_denoised.npy'
    np.save(data_fname, denoised_data)

    confound_fname = os.path.join(os.getcwd(), "all_confounds.txt")
    np.savetxt(confound_fname, confounds, fmt="%.10f")

    return os.path.abspath(img_fname), os.path.abspath(data_fname), confound_fname


def weighted_avg(in_file, mask_file=None):
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import split_filename

    img = nb.load(in_file)
    data = img.get_data()
    weighted_data = np.zeros(data.shape[:3])
    for d in range(data.shape[3]):
        weighted_data += data[:,:,:,d]*np.square(d+1)
    weighted_data /= data.shape[3]

    if mask_file is not None:
        weighted_data *= nb.load(mask_file).get_data()

    _, base, _ = split_filename(in_file)
    fname = base + '_weighted.nii.gz'
    nb.save(nb.Nifti1Image(weighted_data, img.affine, img.header), fname)

    return os.path.abspath(fname)


'''
======================================
Functions copied from Nipype workflows
======================================
'''

def selectindex(files, idx):
    import numpy as np
    from nipype.utils.filemanip import filename_to_list, list_to_filename
    return list_to_filename(np.array(filename_to_list(files))[idx].tolist())

def median(in_files):
    """Computes an average of the median of each realigned timeseries
    Parameters
    ----------
    in_files: one or more realigned Nifti 4D time series
    Returns
    -------
    out_file: a 3D Nifti file
    """
    import nibabel as nb
    import numpy as np
    import os
    from nipype.utils.filemanip import filename_to_list
    from nipype.utils.filemanip import split_filename

    average = None
    for idx, filename in enumerate(filename_to_list(in_files)):
        img = nb.load(filename)
        data = np.median(img.get_data(), axis=3)
        if average is None:
            average = data
        else:
            average = average + data
    median_img = nb.Nifti1Image(average/float(idx + 1),
                                img.get_affine(), img.get_header())
    #filename = os.path.join(os.getcwd(), 'median.nii.gz')
    #median_img.to_filename(filename)
    _, base, _ = split_filename(filename_to_list(in_files)[0])
    nb.save(median_img, base + "_median.nii.gz")
    return os.path.abspath(base + "_median.nii.gz")
    return filename


def strip_rois_func(in_file, t_min):
    import numpy as np
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    nii = nb.load(in_file)
    new_nii = nb.Nifti1Image(nii.get_data()[:,:,:,t_min:], nii.get_affine(), nii.get_header())
    new_nii.set_data_dtype(np.float32)
    _, base, _ = split_filename(in_file)
    nb.save(new_nii, base + "_roi.nii.gz")
    return os.path.abspath(base + "_roi.nii.gz")


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative
    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    from nipype.utils.filemanip import filename_to_list
    import numpy as np
    import os

    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor_der%d_ord%d.txt" % (derivatives, order))
        np.savetxt(filename, out_params2, fmt="%.10f")
        out_files.append(filename)
    return out_files

'''
===============================================
Functions from Rafael Neto Henriques' pca_utils
(adpapted to input and output files)
===============================================
'''

def pca_noise_classifier(L, m):
    """ Classify which PCA eigenvalues are related to noise

    Parameters
    ----------
    L : array (n,)
        Array containing the PCA eigenvalues.

    Returns
    -------
    c : int
        Number of eigenvalues related to noise
    sig2 : float
        Estimation of the noise variance
    """
    import numpy as np

    sig2 = np.mean(L)
    c = L.size - 1
    r = L[c] - L[0] - 4 * np.sqrt((c+1.0) / m) * sig2
    while r > 0:
        sig2 = np.mean(L[:c])
        c = c - 1
        r = L[c] - L[0] - 4*np.sqrt((c+1.0) / m) * sig2
    return c + 1, sig2


def pca_denoising(in_file, ps=2, overcomplete=True):
    """ Denoises DWI volumes using PCA analysis and Marchenko–Pastur
    probability theory

    Parameters
    ----------
    in_file : 4D image file
    ps : int
        Number of neighbour voxels for the PCA analysis.
        Default: 2
    overcomplete : boolean
        If set to True, overcomplete local PCA is computed
        Default: False

    Returns
    -------
    den : array ([X, Y, Z, g])
        Matrix containing the denoised 4D DWI data.
    std : array ([X, Y, Z])
        Matrix containing the noise std estimated using
        Marchenko-Pastur probability theory.
    ncomps : array ([X, Y, Z])
        Number of eigenvalues preserved for the denoised
        4D data.
    """
    import numpy as np
    import nibabel as nb
    import os
    from nipype.utils.filemanip import split_filename
    from functions import pca_noise_classifier

    # Load data
    img = nb.load(in_file)
    dwi = img.get_data()

    # Compute dimension of neighbour sliding window
    m = (2*ps + 1) ** 3

    n = dwi.shape[3]
    den = np.zeros(dwi.shape)
    ncomps = np.zeros(dwi.shape[:3])
    sig2 = np.zeros(dwi.shape[:3])
    if overcomplete:
        wei = np.zeros(dwi.shape)

    for k in range(ps, dwi.shape[2] - ps):
        for j in range(ps, dwi.shape[1] - ps):
            for i in range(ps, dwi.shape[0] - ps):
                # Compute eigenvalues for sliding window
                X = dwi[i - ps: i + ps + 1, j - ps: j + ps + 1,
                        k - ps: k + ps + 1, :]
                X = X.reshape(m, n)
                M = np.mean(X, axis=0)
                X = X - M
                [L, W] = np.linalg.eigh(np.dot(X.T, X)/m)

                # Find number of noise related eigenvalues
                c, sig = pca_noise_classifier(L, m)

                # Reconstruct signal without noise components
                Y = X.dot(W[:, c:])
                X = Y.dot(W[:, c:].T)
                X = X + M
                X = X.reshape(2*ps + 1, 2*ps + 1, 2*ps + 1, n)

                # Overcomplete weighting
                if overcomplete:
                    w = 1.0 / (1.0 + n - c)
                    wei[i - ps: i + ps + 1,
                        j - ps: j + ps + 1,
                        k - ps: k + ps + 1, :] = wei[i - ps: i + ps + 1,
                                                     j - ps: j + ps + 1,
                                                     k - ps: k + ps + 1, :] + w
                    X = X * w
                    den[i - ps: i + ps + 1,
                        j - ps: j + ps + 1,
                        k - ps: k + ps + 1, :] = den[i - ps: i + ps + 1,
                                                     j - ps: j + ps + 1,
                                                     k - ps: k + ps + 1, :] + X
                    ncomps[i - ps: i + ps + 1,
                           j - ps: j + ps + 1,
                           k - ps: k + ps + 1] = ncomps[i - ps: i + ps + 1,
                                                        j - ps: j + ps + 1,
                                                        k - ps: k + ps + 1] + (n-c)*w
                    sig2[i - ps: i + ps + 1,
                           j - ps: j + ps + 1,
                           k - ps: k + ps + 1] = sig2[i - ps: i + ps + 1,
                                                      j - ps: j + ps + 1,
                                                      k - ps: k + ps + 1] + sig*w
                else:
                    den[i, j, k, :] = X[ps, ps, ps]
                    ncomps[i, j, k] = n - c
                    sig2[i, j, k] = sig

    if overcomplete:
        den = den / wei
        ncomps = ncomps / wei[..., 0]
        sig2 = sig2 / wei[..., 0]

    _, base, _ = split_filename(in_file)
    den_path = base + '_mp_denoised.nii.gz'
    sig_path = base + '_mp_sigmas.nii.gz'
    comp_path = base + '_mp_comps.nii.gz'
    nb.save(nb.Nifti1Image(den, img.affine, img.header), den_path)
    nb.save(nb.Nifti1Image(np.sqrt(sig2), img.affine, img.header), sig_path)
    nb.save(nb.Nifti1Image(ncomps, img.affine, img.header), comp_path)

    return os.path.abspath(den_path), os.path.abspath(sig_path), os.path.abspath(comp_path)
