import numpy as np

from typing import List, Optional, Tuple, Union
from numpy.typing import NDArray



def compute_vasc_DICO_with_one_vascular_distribution(
        base_DICO_parameters: NDArray[np.float64],
        base_DICO_signals: NDArray[np.complex128],
        distrib_DICO_parameters: NDArray[np.float64],
        distrib_DICO_coefs: NDArray[np.float64],
        id_vascular_parameters: Optional[NDArray[np.int64]] = None, 
        id_df: int = 2,
        keeped_df: Optional[Union[float, List[float]]] = None
        ) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Convolves each signal of a base dictionary with one (potentially not always the same) vascular distribution. 

    Parameters
    ----------
    base_DICO_parameters: 2d array of shape (n_signals, n_parameters)
        Array countaining the dictionary parameters associated to generated signals. 
    base_DICO_signals: 2d array of shape (n_signals, n_pulses)
        Array countaining the base dictionary signals. 
    distrib_DICO_parameters: 2d array of shape (n_distribs, 3)
        Array countaining the vascular parameters values (S02, Vf, R) associated to generated coefficients. 
    distrib_DICO_coefs: 2d array of shape (n_distribs, n_df)
        Array countaining the distribution coefficients. 
    id_vascular_parameters: 1d array of shape (n_signals), optional
        Vascular distribution indexes. If an int is provided, all signals of the base dictionary will be convolved with the same 
        vascular distribution associated to that index. If an array is provided, each line of the base dictionary will be convolved
        with the vascular distribution associated to the corresponding index in the list. By default, the function will draw a random 
        list of distirbution indexes. 
    id_df: int
        Index of the 'df' parameter into DICO_parameters. 
    keeped_df: float or Lst[float], optional
        If a list of two floats is provided, only keep signals for which the df parameter belongs to the range given by the list. If a float 
        is provided, the same holds in the symetrical range [-keeped_df, keeped_df]. The default range is [df_min+20, df_max-20] where df_min 
        and df_max are the minimal and maximal values of the df parameters in the base dictionary (to avoid unrealistic distributions). 

    Returns
    -------
    vasc_DICO_parameters: 2d array of shape (n_signals2, n_parameters2)
        Array countaining the dictionary parameters associated to generated signals. 
    vasc_DICO_signals: 2d array of shape (n_signals2, n_pulses2)
        Array countaining the convolved dictionary signals. 
    """
    n_signals, n_pulses = base_DICO_signals.shape
    n_distribs, n_df_coefs = distrib_DICO_coefs.shape

    lst_df = np.unique(base_DICO_parameters[:, id_df])
    n_df = lst_df.shape[0]
    df_min, df_max = min(lst_df), max(lst_df)
    
    if n_df != n_df_coefs:
        raise ValueError("Number of df values in the base ditionary ({}) does not correspond with number of df bins in vascular distributions ({})".format(n_df, n_df_coefs))
    
    # compute df values of signals keeped in the convolved dictionary
    df0_min, df0_max = _compute_keeped_df_range(keeped_df, lst_df)
    lst_df0 = lst_df[(df0_min <= lst_df) * (lst_df <= df0_max)]
    n_df0 = lst_df0.shape[0]
    df0_offset = np.argmax(lst_df == lst_df0[0])

    # if no vascular indexes have been provided, choose random vascular indexes
    if id_vascular_parameters is None:
        id_vascular_parameters = np.random.randint(n_distribs, size=n_signals)
    if isinstance(id_vascular_parameters, int):
        id_vascular_parameters = id_vascular_parameters * np.ones(n_signals, dtype=np.int64)
    
    df_coefs_train = distrib_DICO_coefs[id_vascular_parameters]

    # compute indices of signals keeped in the convolved dictionary
    id_keeped_signals = (df0_min <= base_DICO_parameters[:, id_df]) * (base_DICO_parameters[:, id_df] <= df0_max)

    # add vasc parameters to the dictionary and remove lines for which df0 is too small or too big
    vasc_DICO_parameters = np.concatenate((base_DICO_parameters, distrib_DICO_parameters[id_vascular_parameters]), axis=1)
    vasc_DICO_parameters = vasc_DICO_parameters[id_keeped_signals]

    # convolve signals to obtain the dictionary
    vasc_DICO_signals_tensor = base_DICO_signals.reshape((-1, n_df, n_pulses))
    vasc_DICO_signals = np.zeros(
        (vasc_DICO_parameters.shape[0], n_pulses), dtype=np.complex128)
    for (id_df0, df0) in enumerate(lst_df0):
        # convolution coefficients, shape (n_signals//n_df, n_df)
        convol_coefs_df0 = np.zeros((n_signals//n_df, n_df))
        for (id_convol, coefs) in enumerate(df_coefs_train[df0_offset+id_df0::n_df]):
            convol_coefs_df0[id_convol] = np.roll(coefs, int(df0))   # TODO: to be improved to account for df list not in the form df_min:df_max:1
        convol_coefs_df0 /= np.linalg.norm(convol_coefs_df0, ord=1, axis=1, keepdims=True)

        vasc_DICO_signals[id_df0::n_df0] = np.sum(
            vasc_DICO_signals_tensor * convol_coefs_df0[:, :, None], axis=1)

    return vasc_DICO_parameters, vasc_DICO_signals


# internal function utils

def _compute_keeped_df_range(keeped_df: Optional[Union[float, List[float]]],
                             lst_df: NDArray[np.float64],
                             lst_gamma: Union[float, NDArray[np.float64]] = 20.0,
                             ) -> Tuple[float, float]:
    """
    Provides the minimal and maximal df parameter values of signals keeped in the convolved dictionary.

    Parameters
    ----------
    keeped_df: float or Lst[float], optional
        If a list of two floats is provided, the function returns those two floats. If a float is provided, it returns -keeped_df and keeped_df. 
        If None, the function will return df_min+gamma_max and df_max-gamma_max where df_min and df_max are the minimal and maximal values of 
        lst_df and gamma_max is the maximal value of lst_gamma.  
    lst_df: float or 1d array
        List of df values in the base dictionary
    lst_gamma: float of 1d array of shape (n_signals)
        List of gamma values.
    
    Returns
    -------
    df0_min: float
        Minimal df parameter values of signals keeped in the convolved dictionary. 
    df0_max: float
        Maximal df parameter values of signals keeped in the convolved dictionary. 
    """
    if isinstance(keeped_df, list):
        df0_min, df0_max = keeped_df
    elif isinstance(keeped_df, float) or isinstance(keeped_df, int):
        df0_min, df0_max = [-keeped_df, keeped_df]
    else:
        gamma_max = np.max(lst_gamma)
        df_min, df_max = min(lst_df), max(lst_df)
        df0_min, df0_max = df_min + gamma_max, df_max - gamma_max
    
    return df0_min, df0_max
