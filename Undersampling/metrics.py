import numpy as np
from numpy.typing import NDArray
from typing import Tuple

def compare_pre(reconstruction_ref: NDArray, reconstruction_test: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compare two sets of quantitative maps obtained with MRF by computing 
    the voxel-wise relative error (PRE), absolute difference, and total PRE error per parameter.

    Parameters
    ----------
    reconstruction_ref : NDArray
        The reference reconstructed parameter maps with shape (n_params, x, y, z).
    reconstruction_test : NDArray
        The test reconstructed parameter maps with shape (n_params, x, y, z) to compare against the reference.

    Returns
    -------
    pre : NDArray
        The voxel-wise percentage relative error (PRE), computed as:
        100 * (ref - test) / ref. Shape: (n_params, x, y, z).
    diff : NDArray
        The voxel-wise absolute difference: ref - test. Shape: (n_params, x, y, z).
    pre_sum : NDArray
        Sum of the absolute PRE over the full volume for each parameter (length n_params).
        For parameters at indices 2 and 4, the absolute difference is returned instead of PRE.
    """
    for i in range (6):
        reconstruction_ref[i, :, :, :] = np.where(reconstruction_ref[i, :, :, :] == 0, 0.0005, reconstruction_ref[i, :, :, :])

    pre = (reconstruction_ref[:, :, :, :]-reconstruction_test[:, :, :, :])*100/reconstruction_ref[:, :, :, :]
    diff = (reconstruction_ref[:, :, :, :]-reconstruction_test[:, :, :, :])

    reconstruction_test_nan = reconstruction_test.copy()
    reconstruction_ref_nan = reconstruction_ref.copy()

    reconstruction_test_nan[np.isnan(reconstruction_test_nan)] = 1
    reconstruction_ref_nan[np.isnan(reconstruction_ref_nan)] = 1

    pre_sum = np.sum(np.abs(reconstruction_ref_nan[:, :, :, :]-reconstruction_test_nan[:, :, :, :])*100/np.abs(reconstruction_ref_nan[:, :, :, :]), axis=(1,2,3))
    diff_sum = np.sum(np.abs(reconstruction_ref_nan[:, :, :, :]-reconstruction_test_nan[:, :, :, :]), axis=(1,2,3))

    pre_sum[2] = diff_sum[2]
    pre_sum[4] = diff_sum[4]

    return pre, diff, pre_sum