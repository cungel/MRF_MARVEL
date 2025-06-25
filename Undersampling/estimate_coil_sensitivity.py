import numpy as np
from numpy.typing import NDArray
import sys
import os

cg_sense_path = os.path.abspath(os.path.join(os.getcwd(), "CG-SENSE"))
if cg_sense_path not in sys.path:
    sys.path.append(cg_sense_path)

from cg_sense_new_version import * 
import coil as calculate_coil_sens
import CG_signal

def estimate_sensitivity(kspace: NDArray) -> NDArray:
    """
    Estimate the coil sensitivity maps from the k-space data.

    Parameters
    ----------
    kspace : NDArray
        The k-space data. Format: (n_coils, n_pulses, x, y)

    Returns
    -------
    NDArray
        The estimated coil sensitivity maps.
    """
    def filter_fn(x):
        return CG_signal.hann(2 * x)
    
    filtered_kspace = CG_signal.filter_kspace(kspace[:,0,:,:],filter_fn=filter_fn, filter_rank=2)

    img_lowres = np.fft.fftshift( np.fft.ifft2(np.fft.fftshift(filtered_kspace, axes=(-2, -1))), axes=(-2, -1))

    sensitivities = np.zeros_like(img_lowres)
    sensitivities = calculate_coil_sens.estimate_coil_sensitivities(img_lowres, coil_axis=0, method='walsh')

    return sensitivities