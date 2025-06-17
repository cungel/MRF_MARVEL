##########################
## RECONSTRUCTION TOOLS ##
##########################


import os
import sys

import numpy as np

from typing import List, Optional, Tuple
from numpy.typing import NDArray
from tensorflow.keras import Model

from sklearn.model_selection import train_test_split

from .load_save_utils import load_parameter_names_units_limits
_, _, PARAMETER_LIMS_BRAIN, PARAMETER_LIMS_DICO, _ = load_parameter_names_units_limits(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))



# Preprocess
############

def compute_train_test_data(DICO_normalized_signals: NDArray, 
                            DICO_parameters: NDArray, 
                            label_parameters: List[str], 
                            normalize: bool = True, 
                            test_size: float = 0.1, 
                            random_state: int = 0
                            ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """TODO"""
    x = np.copy(DICO_normalized_signals)
    if normalize: 
        y = normalize_params(DICO_parameters, label_parameters)
    else:
        y = np.copy(DICO_parameters)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test

def normalize_params(DICO_parameters: NDArray, 
                     label_parameters: List[str]
                     ) -> NDArray: 
    """TODO"""

    LIMS_DICO = np.array([PARAMETER_LIMS_DICO[param] for param in label_parameters])

    return (DICO_parameters - LIMS_DICO[None, :, 0]) / (LIMS_DICO[None, :, 1] - LIMS_DICO[None, :, 0])
    

# Maps Reconstruction
#########################


def NN_prediction(NN: Model, 
                  acquisition: NDArray, 
                  label_parameters: List[str], 
                  roi_map: Optional[NDArray] = None, 
                  batch_size=64, 
                  normalize=True, 
                  postprocess: bool = True
                  ) -> NDArray:
    """#TODO"""

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
    """#TODO"""

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
