import os
import sys

import numpy as np

from typing import List, Optional, Tuple
from numpy.typing import NDArray
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split

from .load_save_utils import load_parameter_names_units_limits
_, _, PARAMETER_LIMS_BRAIN, PARAMETER_LIMS_DICO, _ = load_parameter_names_units_limits(os.path.join(os.path.dirname(__file__), os.pardir))


# Preprocessing

def compute_train_test_data(DICO_normalized_signals: NDArray, 
                            DICO_parameters: NDArray, 
                            label_parameters: List[str], 
                            normalize: bool = True, 
                            test_size: float = 0.1, 
                            random_state: int = 0
                            ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Split and normalize training and testing datasets from a signal dictionary and associated parameters.

    Parameters
    ----------
    DICO_normalized_signals : NDArray
        Normalized signal dictionary.
    DICO_parameters : NDArray
        Corresponding physical parameters for each signal.
    label_parameters : List[str]
        List of parameter names.
    normalize : bool, optional
        Whether to normalize the parameters (default is True).
    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.1).
    random_state : int, optional
        Seed for reproducibility (default is 0).

    Returns
    -------
    x_train : NDArray
        Training signals.
    x_test : NDArray
        Testing signals.
    y_train : NDArray
        Training parameters (normalized if normalize=True).
    y_test : NDArray
        Testing parameters (normalized if normalize=True).
    """
    x = np.copy(DICO_normalized_signals)

    if normalize: 
        y = normalize_params(DICO_parameters, label_parameters)
    else:
        y = np.copy(DICO_parameters)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test


def normalize_params(DICO_parameters: NDArray, 
                     label_parameters: List[str]
                     ) -> NDArray: 
    """
    Normalizes parameter values to the [0, 1] range using predefined limits for each parameter.

    Parameters
    ----------
    DICO_parameters : NDArray
        Array of parameter values for each signal.
        The order of parameters must match the order in `label_parameters`.
    label_parameters : List[str]
        List of parameter names to normalize. Each must be a key in PARAMETER_LIMS_DICO.

    Returns
    -------
    NDArray
        Normalized parameters in the range [0, 1].

    """
    LIMS_DICO = np.array([PARAMETER_LIMS_DICO[param] for param in label_parameters])
    return (DICO_parameters - LIMS_DICO[None, :, 0]) / (LIMS_DICO[None, :, 1] - LIMS_DICO[None, :, 0])
    

# Maps Reconstruction

def NN_prediction(NN: Model, 
                  acquisition: NDArray, 
                  label_parameters: List[str], 
                  roi_map: Optional[NDArray] = None, 
                  batch_size=64, 
                  normalize=True, 
                  postprocess: bool = True
                  ) -> NDArray:
    """
    Applies MARVEL to an acquisition (magnitude only) to estimate parameter maps.

    Parameters
    ----------
    NN : Model
        Trained model (MARVEL) that predicts physical parameters from input signals.
    acquisition : NDArray
        MRF signals with shape (n_x, n_y, n_z, n_pulses).
    label_parameters : List[str]
        List of parameter names to be predicted.
    roi_map : Optional[NDArray], optional
        3D boolean mask of shape (n_x, n_y, n_z) indicating the region of interest (ROI) to predict.
        If None, the whole volume is used.
    batch_size : int, optional
        Number of voxels to predict in parallel. Default is 64.
    normalize : bool, optional
        If True, rescales predicted parameters from [0, 1] back to physical units using `PARAMETER_LIMS_DICO`.
    postprocess : bool, optional
        If True, clips predictions to remain within physiologically plausible bounds using `PARAMETER_LIMS_BRAIN`.

    Returns
    -------
    prediction : NDArray
        4D array of shape (n_parameters, n_x, n_y, n_z) with the predicted parameter maps.
        Voxels outside the ROI are filled with NaNs.
    """
    n_parameters = len(label_parameters)
    n_x, n_y, n_z, n_pulses = acquisition.shape

    if roi_map is None:
        roi_map = np.ones(acquisition.shape[:-1], dtype=bool)

    if batch_size == 0:
        batch_size = np.sum(roi_map)
    
    # make predictions and store it in their associated voxels. 
    prediction = np.zeros((n_parameters, n_x, n_y, n_z))
    prediction[:, roi_map] = np.swapaxes(NN.predict(acquisition[roi_map], batch_size=batch_size), 0, 1)

    # Rescale parameter values from [0, 1] to [param_min, param_max] limit values in the DICO 
    if normalize:
        LIMS_DICO = np.array([PARAMETER_LIMS_DICO[param] for param in label_parameters])
        prediction = LIMS_DICO[:, 0, None, None, None] + prediction * (LIMS_DICO[:, 1] - LIMS_DICO[:, 0])[:, None, None, None]

    # Clip parameter values inside [param_min, param_max] limit values in the BRAIN 
    if postprocess:
        LIMS_BRAIN = np.array([PARAMETER_LIMS_BRAIN[param] for param in label_parameters])
        prediction = np.clip(prediction, LIMS_BRAIN[:, 0, None, None, None], LIMS_BRAIN[:, 1, None, None, None])
    
    prediction[:, ~roi_map] = np.nan

    return prediction

def NN_prediction_complex(NN: Model, 
                  acquisition: NDArray, 
                  label_parameters: List[str], 
                  roi_map: Optional[NDArray] = None, 
                  batch_size=64, 
                  normalize=True, 
                  postprocess: bool = True
                  ) -> NDArray:
    """
    Applies MARVEL to a complex acquisition to estimate parameter maps.

    Parameters
    ----------
    NN : Model
        Trained model (MARVEL) that predicts physical parameters from input signals.
    acquisition : NDArray
        MRF complex signals with shape (n_x, n_y, n_z, n_pulses).
    label_parameters : List[str]
        List of parameter names to be predicted.
    roi_map : Optional[NDArray], optional
        3D boolean mask of shape (n_x, n_y, n_z) indicating the region of interest (ROI) to predict.
        If None, the whole volume is used.
    batch_size : int, optional
        Number of voxels to predict in parallel. Default is 64.
    normalize : bool, optional
        If True, rescales predicted parameters from [0, 1] back to physical units using `PARAMETER_LIMS_DICO`.
    postprocess : bool, optional
        If True, clips predictions to remain within physiologically plausible bounds using `PARAMETER_LIMS_BRAIN`.

    Returns
    -------
    prediction : NDArray
        4D array of shape (n_parameters, n_x, n_y, n_z) with the predicted parameter maps.
        Voxels outside the ROI are filled with NaNs.
    """
    n_parameters = len(label_parameters)
    n_x, n_y, n_z, n_pulses = acquisition.shape

    if roi_map is None:
        roi_map = np.ones(acquisition.shape[:-1], dtype=bool)

    if batch_size == 0:
        batch_size = np.sum(roi_map)
    
    # make predictions and store it in their associated voxels. 
    prediction = np.zeros((n_parameters, n_x, n_y, n_z))
    prediction[:, roi_map] = np.swapaxes(NN.predict(np.concatenate((np.abs(acquisition), np.unwrap(np.angle(acquisition))), axis=-1)[roi_map], batch_size=batch_size), 0, 1)

    # Rescale parameter values from [0, 1] to [param_min, param_max] limit values in the DICO 
    if normalize:
        LIMS_DICO = np.array([PARAMETER_LIMS_DICO[param] for param in label_parameters])
        prediction = LIMS_DICO[:, 0, None, None, None] + prediction * (LIMS_DICO[:, 1] - LIMS_DICO[:, 0])[:, None, None, None]

    # Clip parameter values inside [param_min, param_max] limit values in the BRAIN 
    if postprocess:
        LIMS_BRAIN = np.array([PARAMETER_LIMS_BRAIN[param] for param in label_parameters])
        prediction = np.clip(prediction, LIMS_BRAIN[:, 0, None, None, None], LIMS_BRAIN[:, 1, None, None, None])
    
    prediction[:, ~roi_map] = np.nan

    return prediction
