import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union

def cartesian_undersampling(RECO: NDArray,
                            percentage_undersampling: float,
                            return_ksp: bool = True
                            ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    Apply a cartesian undersampling mask to the data.
    
    Parameters
    ----------
    RECO: NDArray
        The image to undersample of shape (x, y, z, t).
    percentage_undersampling: float
        Percentage of undersampling to apply.
    return_ksp: bool
        If True, returns the k-space undersampled data.
    
    Returns
    -------
    inverse_fft: NDArray
        The undersampled image in the spatial domain.
    k_space_undersampled: NDArray, optional
        The k-space undersampled data if return_ksp is True.
    """
    k_space = np.fft.fftshift(np.fft.fft2(RECO, axes=(0,1)), axes=(0, 1))
    k_space_undersampled = k_space.copy()

    nb_del_lines = int(np.round((percentage_undersampling * k_space.shape[0]/100)))

    if nb_del_lines % 2 == 0 :
        for i in range (int(nb_del_lines/2)) :
            k_space_undersampled[i,:,:,:] = 0
            k_space_undersampled[-1-i,:,:,:] = 0
    else:
        for i in range (int((nb_del_lines+1)/2)) :
            k_space_undersampled[i,:,:,:] = 0
            k_space_undersampled[-i,:,:,:] = 0

    inverse_fft = np.fft.ifft2(np.fft.ifftshift(k_space_undersampled, axes=(0, 1)), axes = (0,1))

    if return_ksp:
        return inverse_fft, k_space_undersampled
    else:
        return inverse_fft