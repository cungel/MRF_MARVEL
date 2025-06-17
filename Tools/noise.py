import numpy as np

from typing import List,Union
from numpy.typing import NDArray

def add_Gaussian_noise_to_DICO(DICO_signals: NDArray[np.float64],
                               SNR: Union[float, List[float]],
                               SNR_type: str = 'poisson'
                               ) -> NDArray[np.float64]:

    if SNR_type == 'poisson':
        SNR = np.random.poisson(SNR, DICO_signals.shape[0])[:, None]
        SNR[SNR == 0] = 1

    if SNR_type == 'uniform':
        SNR = SNR[0] + (SNR[1] - SNR[0]) * np.random.rand(DICO_signals.shape[0])[:, None]

    noise = np.random.randn(*DICO_signals.shape)
    noise = noise * np.mean(DICO_signals) / SNR
    magnitude_noisy = DICO_signals + noise
 
    return magnitude_noisy

def add_Gaussian_noise_to_DICO_complex(DICO_signals: NDArray[np.complex128],
                               SNR: Union[float, List[float]],
                               SNR_type: str = 'uniform'
                               ) -> NDArray[np.complex128]:
    """
    Add Gaussian noise to DICO signals (complex signals).

    Parameters
    ----------
    DICO_signals : NDArray[np.complex128]
        The DICO signals (complex values).
    SNR : Union[float, List[float]]
        Signal-to-noise ratio.
    SNR_type : str, optional
        NOT IMPLEMENTED.

    Returns
    -------
    NDArray[np.complex128]
        The noisy DICO signals (complex).
    """
    magnitude = np.abs(DICO_signals)
    phase = np.angle(DICO_signals)

    SNR_magnitude = SNR[0] + (SNR[1] - SNR[0]) * np.random.rand(DICO_signals.shape[0])[:, None]
    SNR_phase = 5 + (20- 5) * np.random.rand(DICO_signals.shape[0])[:, None]

    noise = np.random.randn(*DICO_signals.shape)
    noise_magnitude = noise * np.mean(magnitude) / SNR_magnitude
    magnitude_noisy = magnitude + noise_magnitude

    noise_phase = np.random.randn(*DICO_signals.shape)
    noise_phase = noise_phase * (1.0 / SNR_phase)  
    phase_noisy = phase + noise_phase

    return magnitude_noisy * np.exp(1j * phase_noisy)