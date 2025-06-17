import os
import re
import json
import scipy.io as sio
import mat73
import h5py
import nibabel as nib      
import numpy as np
from typing import List, Tuple, Optional, Any
from numpy.typing import NDArray
from tensorflow.keras import Model
from scipy.ndimage import gaussian_filter

# Dictionnaries

def load_matlab_dictionary(
        path_to_dico: str, 
        v: bool = True, 
        nb_indents: int = 0
        ) -> Tuple[NDArray[np.float64], NDArray[np.complex64], List[str]]:
    """ 
    Load a matlab dictionary. 

    Parameters
    ----------
    path_to_dico: str
        Path of the Matlab dictionary file. 
    v: bool, optional
        If True, prints information about the load. Default to True. 
    nb_indents: int
        In verbose mode, number of indents to add to the text. Default to 0. 
    
    Returns
    -------
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    DICO_signals: 2d array of shape (n_signals, n_pulses)
        Array countaining the dictionary signals. 
    label_parameters: List[str]
        List of parameter labels. 
    """
    if not path_to_dico.endswith('.mat'):
        path_to_dico += '.mat'

    DICO = mat73.loadmat(path_to_dico)

    # matrix of MRI signals of shape n_signals x n_pulses
    DICO_signals = DICO['Dico']['MRSignals'][0]
    (n_signals, n_pulses) = DICO_signals.shape

    # matrix of parameters values (n_signals x n_parameters) associated to each signal.
    DICO_parameters = DICO['Dico']['Parameters']['Par']
    label_parameters = DICO['Dico']['Parameters']['Labels']
    n_parameters = DICO_parameters.shape[1]

    if v:
        print(('{}Loading of a dictionary of {} signals of length {} pulses. \n{}There are {} parameters: {}.').format('    ' * nb_indents, n_signals, n_pulses, '    ' * nb_indents, n_parameters, ', '.join(label_parameters)))
    
    return DICO_parameters, DICO_signals, label_parameters


def load_hdf5_dictionary(
        path_to_dico: str, 
        v: bool = True, 
        nb_indents: int = 0
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """ 
    Load a hdf5 dictionary. 

    Parameters
    ----------
    path_to_dico: str
        Path of the hdf5 dictionary file. 
    v: bool, optional
        If True, prints information about the load. Default to True. 
    nb_indents: int
        In verbose mode, number of indents to add to the text. Default to 0. 
    
    Returns
    -------
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    DICO_signals: 2d array of shape (n_signals, n_pulses)
        Array countaining the dictionary signals. 
    label_parameters: List[str]
        List of parameter labels. 
    """
    if not path_to_dico.endswith('.h5'):
        path_to_dico += '.h5'

    DICO = h5py.File(path_to_dico, "r")

    # matrix of MRI signals of shape n_signals x n_pulses
    DICO_signals = np.abs(np.array(DICO.get('MRSignals'))[:, :, 0])
    (n_signals, n_pulses) = DICO_signals.shape

    # matrix of parameters values (n_signals x n_parameters) associated to each signal.
    DICO_parameters = np.array(DICO.get('Parameters'))
    n_parameters = DICO_parameters.shape[1]

    if v:
        print(('{}Loading of a dictionary of {} signals of length {} pulses. \n{}There are {} parameters.').format('    ' * nb_indents, n_signals, n_pulses, '    ' * nb_indents, n_parameters))
    
    return DICO_parameters, DICO_signals 


def remove_parameters_from_dictionary(
        lst_param: List[str], 
        DICO_parameters: NDArray[np.float64], 
        label_parameters: List[str], 
        v: bool = True, 
        nb_indents: int = 0, 
        ) -> Tuple[NDArray[np.float64], List[str]]:
    """ 
    Remove parameters from a dictionary. 

    Parameters
    ----------
    lst_param: Lits[str]
        Lists of names of removed parameters. 
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    label_parameters: List[str]
        List of parameter labels. 
    v: bool, optional
        If true, prints information about the load. Default to True. 
    nb_indents: int
        In verbose mode, number of indents to add to the text. Default to 0. 

    Returns
    -------
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the modified dictionary parameters. 
    label_parameters: List[str]
        List of parameter labels. 
    """
    n_parameters = DICO_parameters.shape[1]

    unfound_params = 0
    for param in lst_param:
        try:
            eliminated_index = label_parameters.index(param)
        except:
            print('    ' * nb_indents + "Error: parameter {} not found.".format(param))
            unfound_params += 1
        else:
            DICO_parameters = np.delete(DICO_parameters, eliminated_index, axis=1) 
            label_parameters.remove(param)

    if v:
        n_parameters_removed = len(lst_param) - unfound_params
        n_parameters = DICO_parameters.shape[1]
        print('    ' * nb_indents + '{} parameter{} removed. There are now {} parameters: {}.'.format(n_parameters_removed, 's' * (n_parameters_removed>1), n_parameters, ', '.join(label_parameters)))
    
    return DICO_parameters, label_parameters


def load_numpy_dictionary(
        path_to_data: str, 
        v: bool = True
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[str]]:
    """ 
    Load a numpy dictionary. 

    Parameters
    ----------
    path_to_data: str
        Path to the 'parameters.npy' and 'signals.npy' files. 
    v: bool, optional
        If true, prints information about the load. Default to True. 
    
    Returns
    -------
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    DICO_signals: 2d array of shape (n_signals, n_pulses)
        Array countaining the dictionary signals. 
    """
    # matrix of MRI signals of shape n_signals x n_pulses
    DICO_signals = np.load(os.path.join(path_to_data, 'signals.npy'))
    (n_signals, n_pulses) = DICO_signals.shape

    # matrix of parameters values (n_signals x n_parameters) associated to each signal.
    DICO_parameters = np.load(os.path.join(path_to_data, 'parameters.npy'))
    n_parameters = DICO_parameters.shape[1]

    if v:
        print(('Loading of a dictionary of {} signals of length {} pulses. \nThere are {} parameters. ').format(n_signals, n_pulses, n_parameters))
    
    return DICO_parameters, DICO_signals


def save_numpy_dictionary(
        path_to_data: str, 
        DICO_parameters: NDArray[np.float64], 
        DICO_signals: NDArray[np.float64]
        ) -> None:
    """ 
    Save a numpy dictionary. 

    Parameters
    ----------
    path_to_data: str
        Path to the 'parameters.npy' and 'signals.npy' files. 
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    DICO_signals: 2d array of shape (n_signals, n_pulses)
        Array countaining the dictionary signals. 
    """
    np.save(os.path.join(path_to_data, 'DICO_parameters.npy'), DICO_parameters)
    np.save(os.path.join(path_to_data, 'DICO_signals.npy'), DICO_signals)


# Distributions

def load_distrib_DICO_from_directory(distribution_path: str) -> Tuple[NDArray, NDArray]:
    """ 
    Load distribution parameters and coefficients of vascular voxels stored in a directory. 

    Parameters
    ----------
    distribution_path: str
        Distributions directory path. 

    Returns
    -------
    distrib_DICO_parameters: 2d array of shape (n_distribs, 3)
        Array countaining the vascular parameters values (S02, Vf, R) associated to generated coefficients. 
    distrib_DICO_coefs: 2d array of shape (n_distribs, n_df)
        Array countaining the distribution coefficients. 
    """
    # get list of distribution files, and remove those with NaN values
    list_distrib_filenames = sorted(os.listdir(distribution_path))
    for i in range(len(list_distrib_filenames)-1, -1, -1):
        if 'NaN' in list_distrib_filenames[i]:
            list_distrib_filenames = list_distrib_filenames[:i] + \
                list_distrib_filenames[i+1:]
    n_distribs = len(list_distrib_filenames)

    # load dictionary parameters of distributions
    distrib_DICO_parameters = np.zeros((n_distribs, 3))
    id_keeped_params = np.ones(n_distribs, dtype=bool)
    for (id_param, distrib_filename) in enumerate(list_distrib_filenames):
        distrib_filepath = os.path.join(distribution_path, distrib_filename)
        # get S02, VF, R values
        SO2 = float(re.search(r'SO2_([0-9.e-]+)', distrib_filepath[:-4]).group(1))
        VF = float(re.search(r'VF_([0-9.e-]+)', distrib_filepath[:-4]).group(1))
        R = float(re.search(r'R_([0-9.e-]+)(?!\d|\.)', distrib_filepath[:-4]).group(1))
        distrib_DICO_parameters[id_param] = SO2, VF, R
        # remove distrib if either S02 or VF or R is 0
        if SO2 == 0 and VF == 0 and R == 0:
            id_keeped_params[id_param] = False
    # keep distributions with correct parameters
    distrib_DICO_parameters = distrib_DICO_parameters[id_keeped_params]
    # put SO2 and Vf in % and R in µm
    distrib_DICO_parameters[:, 0] *= 1e2
    distrib_DICO_parameters[:, 1] *= 1e2
    distrib_DICO_parameters[:, 2] *= 1e6

    # load dictionary coefficients of distributions
    distrib = sio.loadmat(distrib_filepath)
    n_df = distrib['Histo'][0][0][0][0].shape[0]
    distrib_DICO_coefs = np.zeros((n_distribs, n_df))
    for (id_param, distrib_filename) in enumerate(list_distrib_filenames):
        distrib_filepath = os.path.join(distribution_path, distrib_filename)
        distrib = sio.loadmat(distrib_filepath)
        distrib_DICO_coefs[id_param] = distrib['Histo'][0][0][0][0]
    # keep coefficients with correct parameters
    distrib_DICO_coefs = distrib_DICO_coefs[id_keeped_params]

    return distrib_DICO_parameters, distrib_DICO_coefs


def load_distrib_DICO_from_mat(distribution_path: str) -> Tuple[NDArray, NDArray]:
    """ 
    Load distribution parameters and coefficients of vascular voxels stored in a Matlab file. 

    Parameters
    ----------
    distribution_path: str
        Path to Matlab distributions file. 

    Returns
    -------
    distrib_DICO_parameters: 2d array of shape (n_distribs, 3)
        Array countaining the vascular parameters values (S02, Vf, R) associated to generated coefficients. 
    distrib_DICO_coefs: 2d array of shape (n_distribs, n_df)
        Array countaining the distribution coefficients. 
    """
    if not distribution_path.endswith(".mat"):
        distribution_path += ".mat"
    
    #depending on the Matlab file format
    try:
        distribs = mat73.loadmat(distribution_path)['bigStructure']
    except:
        distribs = _loadmat(distribution_path)['bigStructure']

    distrib_DICO_parameters = []
    distrib_DICO_coefs = []
    for key in distribs.keys():
        SO2, VF, R = distribs[key]['SO2'], distribs[key]['VF'], distribs[key]['R']
        coefs = distribs[key]['histo']['bin']

        if not np.any(np.isnan(np.array([SO2, VF, R]))) and not (SO2 == 0.0 and VF == 0.0 and R == 0.0):
            distrib_DICO_parameters.append([SO2, VF, R])
            distrib_DICO_coefs.append(coefs)

    distrib_DICO_parameters = np.asarray(distrib_DICO_parameters)
    distrib_DICO_coefs = np.asarray(distrib_DICO_coefs)

    # put SO2 and Vf in % and R already in µm
    distrib_DICO_parameters[:, 0] *= 1e2
    distrib_DICO_parameters[:, 1] *= 1e2

    return distrib_DICO_parameters, distrib_DICO_coefs


def _loadmat(filename):
    """
    This function should be called instead of direct sio.loadmat
    as it cures the problem of not properly recovering Python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects. 
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dico):
    """
    Checks if entries in dictionary are mat-objects. 
    If yes, _todict is called to change them to nested dictionaries.
    """
    for key in dico:
        if isinstance(dico[key], sio.matlab.mat_struct):
            dico[key] = _todict(dico[key])
    return dico


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    dico = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mat_struct):
            dico[strg] = _todict(elem)
        else:
            dico[strg] = elem
    return dico


# Parameter values

def load_parameter_names_units_limits(path_to_MRF_stroke: str) -> Tuple[dict, dict, dict, dict, dict]:
    """ 
    Load parameter limits and units. 

    Parameter
    ---------
    path_to_MRF_stroke: str
        Path to MRF_stroke repository. 
    
    Returns
    -------
    PARAMETER_DISPLAYED_LABELS: dict
        Dictionary with parameter labels to display on figures. 
    PARAMETER_UNITS: dict
        Dictionary with parameter units. 
    PARAMETER_LIMS_BRAIN: dict
        Dictionary with parameter limits of the brain. 
    PARAMETER_LIMS_DICO: dict
        Dictionary with parameter limits of the dictionary. 
    PARAMETER_LIMS_COLORMAP: dict
        Dictionary with parameter limits of the colormaps.
    """
    
    PARAMETER_LIMS_UNITS = json.load(open(os.path.join(path_to_MRF_stroke, 'Reconstruction/Plot/parameter_limits_and_units.json')))

    PARAMETER_DISPLAYED_LABELS = PARAMETER_LIMS_UNITS['displayed_labels']
    PARAMETER_UNITS = PARAMETER_LIMS_UNITS['units']
    PARAMETER_LIMS_BRAIN = PARAMETER_LIMS_UNITS['brain_lims']
    PARAMETER_LIMS_DICO = PARAMETER_LIMS_UNITS['dico_lims']
    PARAMETER_LIMS_COLORMAP = PARAMETER_LIMS_UNITS['colormap_lims']

    return PARAMETER_DISPLAYED_LABELS, PARAMETER_UNITS, PARAMETER_LIMS_BRAIN, PARAMETER_LIMS_DICO, PARAMETER_LIMS_COLORMAP


def save_parameter_limits_and_units(
        PARAMETER_DISPLAYED_LABELS: dict, 
        PARAMETER_UNITS: dict, 
        PARAMETER_LIMS_BRAIN: dict, 
        PARAMETER_LIMS_DICO: dict, 
        PARAMETER_LIMS_COLORMAP: dict, 
        path_to_MRF_stroke: str
        ) -> None:
    """ 
    Save parameter limits and units to .json file. 

    Parameters
    ----------
    PARAMETER_DISPLAYED_LABELS: dict
        Dictionary with parameter labels to display on figures. 
    PARAMETER_UNITS: dict
        Dictionary with parameter units. 
    PARAMETER_LIMS_BRAIN: dict
        Dictionary with parameter limits of the brain. 
    PARAMETER_LIMS_DICO: dict
        Dictionary with parameter limits of the dictionary. 
    PARAMETER_LIMS_COLORMAP: dict
        Dictionary with parameter limits of the colormaps.
    path_to_MRF_stroke: str
        Path to MRF_stroke repository. 
    """

    PARAMETER_LIMS_UNITS = {
        'displayed_labels': PARAMETER_DISPLAYED_LABELS, 
        'units': PARAMETER_UNITS, 
        'brain_lims': PARAMETER_LIMS_BRAIN, 
        'dico_lims': PARAMETER_LIMS_DICO, 
        'colormap_lims': PARAMETER_LIMS_COLORMAP
    }    

    with open(os.path.join(path_to_MRF_stroke, 'Reconstruction/Plot/parameter_limits_and_units.json'), 'w') as outfile:
        json.dump(PARAMETER_LIMS_UNITS, outfile, indent=4)


def change_parameter_dictionary_limits(
        DICO_parameters: NDArray[np.float64], 
        label_parameters: List[str], 
        path_to_MRF_stroke: str
        ) -> dict:
    """ 
    Update and save parameter limits of the dictionary. 

    Parameters
    ----------
    DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    label_parameters: List[str]
        List of parameter labels. 
    path_to_MRF_stroke: str
        Path to MRF_stroke repository. 
    """

    PARAMETER_DISPLAYED_LABELS, PARAMETER_UNITS, PARAMETER_LIMS_BRAIN, PARAMETER_LIMS_DICO, PARAMETER_LIMS_COLORMAP = load_parameter_names_units_limits(path_to_MRF_stroke)

    for (id_param, param) in enumerate(label_parameters): 
        PARAMETER_LIMS_DICO[param] = [np.min(DICO_parameters[:, id_param]), np.max(DICO_parameters[:, id_param])]
    
    save_parameter_limits_and_units(PARAMETER_DISPLAYED_LABELS, PARAMETER_UNITS, PARAMETER_LIMS_BRAIN, PARAMETER_LIMS_DICO, PARAMETER_LIMS_COLORMAP, path_to_MRF_stroke)

    return PARAMETER_LIMS_DICO


# Acquisitions and reconstructions

def load_masked_acquisition(
        path_to_acquisition: str, 
        rot: int = 0, 
        crop_type: Optional[str] = 'separate'
        ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """ 
    Load a masked acquisition. 
    
    Parameters
    ----------
    path_to_acquisition: str
        Path to '.nii' acquisition file. 
    rot: int
        Rotation angle for the acquisition. Possible values are 0 (default), 90, 180 and 270.  
    crop_type: Optional[str]
        Type of croping of the image. Possibilities are 
        - 'separate': n_x and n_y shapes are independently minimized while keeping the whole brain in the acquisition. 
        - 'equal': n_x and n_y shapes are EQUALLY (n_x = n_y) minimized while keeping the whole brain in the acquisition. 
        - None: no cropping, original n_x and n_y are kept. 

    Returns
    -------
    acquisition: 4d array of shape (n_x, n_y, n_z, n_pulses)
        Array countaining acquisition signals. 
    roi_map: 3d array of shape (n_x, n_y, n_z)
        Boolean array indicating voxels that are in the region of interest, i.e., the brain (True) or outside (False). 
    """
    acquisition = np.array(nib.load(path_to_acquisition).get_fdata())

    for _ in range(rot//90):
        acquisition = np.rot90(acquisition)

    roi_map = ~np.isnan(acquisition[:, :, :, 0])
    
    if crop_type is not None:
        x_min, x_max, y_min, y_max = _get_brain_acquisition_limits(roi_map, crop_type=crop_type)
        acquisition = acquisition[x_min:x_max+1, y_min:y_max+1]
        roi_map = roi_map[x_min:x_max+1, y_min:y_max+1]

    return acquisition, roi_map


def load_acquisition_reconstruction_from_nii(
        file_path: str, 
        rot: int = 0, 
        v: bool = True, 
        nb_indents: int = 0
        ) -> Tuple[NDArray, Any, Any]:
    """ 
    Load reconstruction of an acquisition. 

    Parameters
    ----------
    file_path: str
        Path to the reconstruction file (with or without nii extension). 
    rot: int
        Rotation angle for the acquisition. Possible values are 0 (default), 90, 180 and 270. Warning: only reconstruction 
        array will be rotated, original affine and header will be kept.  
    v: bool
        Verbose mode. Default to True. 
    nb_indents: int
        In verbose mode, number of indents to add to the text. Default to 0. 
    
    Returns
    -------
    reconstruction: Array of shape (n_x, n_y, n_slices, n_pulses)
        Array countaining Philips reconstruction. 
    affine: 
        Metadata associated to the nii file. 
    header: 
        Metadata associated to the nii file. 
    """
    if not file_path.endswith('.nii') and not file_path.endswith('.PAR'):
        file_path += '.nii'

    nii_file = nib.load(file_path)

    reconstruction = np.array(nib.load(file_path).get_fdata())
    affine = nii_file.affine
    header = nii_file.header
    
    for _ in range(rot//90):
        reconstruction = np.rot90(reconstruction)

    if v:
        (n_x, n_y, n_slices, n_pulses) = reconstruction.shape
        print('    ' * nb_indents + 'Loaded a {}x{} reconstruction with {} slice{} and {} pulse{}. '.format(n_x, n_y, n_slices, 's' * (n_slices > 1), n_pulses, 's' * (n_pulses > 1)))
    
    return reconstruction, affine, header


def save_reconstruction_to_nii(
        reconstruction: NDArray[np.float64], 
        affine: Any, 
        header: Any, 
        file_path: str):
    """
    Save Philips reconstruction. 

    Parameters
    ----------
    reconstruction: Array of shape (n_x, n_y, n_slices, n_pulses)
        Array countaining Philips reconstruction. 
    affine: 
        Metadata associated to the nii file. 
    header: 
        Metadata associated to the nii file. 
    file_path: str
        Saving path. 
    """
    nib.save(nib.Nifti1Image(reconstruction, affine, header), file_path)


def load_matching_reconstruction(
        path_to_reconstruction: str, 
        label_parameters: List[str], 
        rot: int = 0, 
        roi_map: Optional[NDArray] = None, 
        path_to_acquisition: Optional[str] = None, 
        crop_type: Optional[str] = 'separate'
        ) -> NDArray[np.float64]:
    """ 
    Load a masked acquisition. 
    
    Parameters
    ----------
    path_to_reconstruction: str
        Path to '.nii' reconstruction files. 
    label_parameters: List[str]
        List of parameter labels. 
    rot: int
        Rotation angle for the acquisition. Possible values are 0 (default), 90, 180 and 270. 
    roi_map: 3d array of shape (n_x, n_y, n_z)
        Boolean array indicating voxels that are in the region of interest, i.e., the brain (True) or outside (False). 
        If unspecified, roi_map will be compiled using path_to_acquisition. 
    path_to_acquisition: str, optional
        Path to '.nii' acquisition file.  Used to compile roi_map if unspecified. If both roi_map and path_to_acquisition 
        are unspecified, voxels outside brain will not be set to np.nan
    crop_type: str; optional
        Type of croping of the image. Possibilities are 
        - 'separate': n_x and n_y shapes are independently minimized while keeping the whole brain in the acquisition. 
        - 'equal': n_x and n_y shapes are EQUALLY (n_x = n_y) minimized while keeping the whole brain in the acquisition. 
        - None: no cropping, original n_x and n_y are kept. 

    Returns
    -------
    reconstruction: 4d array of shape (n_parameters, n_x, n_y, n_z)
        Array countaining reconstructed parameter maps. 
    """
    if roi_map is None and path_to_acquisition is not None:
        _, roi_map = load_masked_acquisition(path_to_acquisition, rot=rot, crop_type=None)

    if roi_map is not None:
        reconstruction = np.zeros((len(label_parameters),)+roi_map.shape)
    else:
        reconstruction_param = np.array(nib.load(path_to_reconstruction + label_parameters[0] + '.nii').get_fdata())
        for _ in range(rot//90):
            reconstruction_param = np.rot90(reconstruction_param)

        reconstruction = np.zeros((len(label_parameters), reconstruction_param.shape))

    for (id_param, param) in enumerate(label_parameters):
        reconstruction_param = np.array(nib.load(path_to_reconstruction + param + '.nii').get_fdata())
        for _ in range(rot//90):
            reconstruction_param = np.rot90(reconstruction_param)
        
        reconstruction[id_param] = reconstruction_param

    if roi_map is not None:
        reconstruction[:, ~roi_map] = np.nan
    
    if crop_type is not None:
        x_min, x_max, y_min, y_max = _get_brain_acquisition_limits(roi_map, crop_type=crop_type)
        reconstruction = reconstruction[:, x_min:x_max+1, y_min:y_max+1]
        roi_map = roi_map[x_min:x_max+1, y_min:y_max+1]

    return reconstruction


def average_signal_pulses(
        signals: NDArray, 
        reducing_factor: int = 3
        ) -> NDArray:
    """ 
    Averages all pulses echoes of a dictionary of signals. 
    
    Parameters
    ----------
    signals: 2d array of shape (n_signals, n_pulses)
        Array countaining signals. 
    reducing_factor: int
        Length of echoes/pulses to average (typically number of echoes per pulse). Default to 3. 

    Returns
    -------
    average_signals: 2d array of shape (n_parameters, n_pulses // reducing_factor)
        Array countaining averaged signals. 
    """
    (n_signals, n_pulses) = signals.shape
    n_pulses //= reducing_factor
    average_signals = np.zeros((n_signals, n_pulses))

    for i in range(reducing_factor):
        average_signals += signals[:, i::reducing_factor]

    average_signals /= reducing_factor

    return average_signals


def average_acquisition_pulses(
        acquisition: NDArray, 
        reducing_factor: int=3
        ) -> NDArray:
    """ 
    Averages all pulses echoes of an acquisition signals. 
    
    Parameters
    ----------
    acquisition: 4d array of shape (n_x, n_y, n_z, n_pulses)
        Array countaining acquisition signals. 
    reducing_factor: int
        Length of echoes/pulses to average (typically number of echoes per pulse). Default to 3. 

    Returns
    -------
    average_acquisition: 2d array of shape (n_parameters, n_pulses // reducing_factor)
        Array countaining averaged acquisition signals. 
    """
    (n_x, n_y, n_z, n_pulses) = acquisition.shape
    n_pulses //= reducing_factor
    average_acquisition = np.zeros((n_x, n_y, n_z, n_pulses))

    for i in range(reducing_factor):
        average_acquisition += acquisition[:, :, :, i::reducing_factor]

    average_acquisition /= reducing_factor

    return average_acquisition


def _get_brain_acquisition_limits(
        roi_map: NDArray[np.bool_], 
        crop_type: str
        ) -> Tuple[int, int, int, int]:
    """ 
    Get the limits of the brain acquisition from a boolean map.

    Parameters
    ----------
    roi_map: 3d array of shape (n_x, n_y, n_z)
        Boolean array indicating voxels that are in the region of interest, i.e., the brain (True) or outside (False). 
    crop_type: str
        Type of croping of the image. Possibilities are 
        - 'separate': n_x and n_y shapes are independently minimized while keeping the whole brain in the acquisition. 
        - 'equal': n_x and n_y shapes are EQUALLY (n_x = n_y) minimized while keeping the whole brain in the acquisition. 

    Returns
    -------
    x_min, x_max, y_min, y_max: int
        Indices such that the whole brain is included in acquisition[x_min:x_max+1, y_min:y_max+1, :], x_max-x_min and y_max-y_min being minimized (and equalized if crop_type is 'equal'). 
    """
    n_x, n_y, _ = roi_map.shape
    
    rows, cols, _ = np.where(roi_map)
    x_min, x_max = min(rows), max(rows)
    y_min, y_max = min(cols), max(cols)

    if crop_type == 'equal':
        diff = abs(x_max-x_min - (y_max-y_min))
        if x_max-x_min > y_max-y_min:
            y_max += diff // 2
            y_min -= diff - diff // 2
            if y_min < 0: 
                y_min, y_max = 0, y_max - y_min
            elif y_max >= n_y:
                y_min, y_max = y_min-y_max+n_y, n_y
        elif y_max-y_min > x_max-x_min:
            x_max += diff // 2
            x_min -= diff - diff // 2
            if x_min < 0: 
                x_min, x_max = 0, x_max - x_min
            elif x_max >= n_x:
                x_min, x_max = x_min-x_max+n_x, n_x

    return x_min, x_max, y_min, y_max


# Networks

def load_model_weights(
        NN: Model, 
        layer_shapes: List[int], 
        day: str = '', 
        adding_text = '', 
        path_to_model: str = '') -> None:
    """ 
    Load weights of a neural network. 

    Parameters
    ----------
    NN: tensorflow model
        Model for which we want to load weights. 
    layer_shapes: List[int]
        List of shapes of the network layers. 
    day: str, optional
        Day of training. If not specified, the date is not added to the filename. 
    adding_text: str, optional
        Optional text to add at the end of the file name. 
    path_to_model: str, optional
        Path to the folder where weights are saved. Default to work directory. 
    """
    model_name = 'model'
    if day != '':
        model_name += '_' + day.replace('-', '')
    for layer_shape in layer_shapes:
        model_name += '_' + str(layer_shape)
    model_name += adding_text + '.h5'

    NN.load_weights(os.path.join(path_to_model, model_name))


def save_model_weights(
        NN: Model, 
        layer_shapes: List[str], 
        day: str = '', 
        adding_text: str = '', 
        path_to_model: str = ''
        ) -> None:
    """ 
    Save weights of a neural network. 

    Parameters
    ----------
    NN: tensorflow model
        Model for which we want to save weights. 
    layer_shapes: List[int]
        List of shapes of the network layers. 
    day: str, optional
        Day of training. Default to today. 
    adding_text: str, optional
        Optional text at the end of the file name. 
    path_to_model: str, optional
        Path to the folder where weights are saved. Default to work directory. 
    """
    model_name = 'model'
    if day != '':
        model_name += '_' + day.replace('-', '')

    for layer_shape in layer_shapes:
        model_name += '_' + str(layer_shape)
    model_name += adding_text + '.h5'

    NN.save(os.path.join(path_to_model, model_name))
