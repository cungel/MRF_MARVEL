import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
from numpy.typing import NDArray
from typing import Tuple, Optional, Union
import pynufft
import warnings

from CG_SENSE.cg_sense_new_version import reconstruct_lstsq


def generate_radiale_traj(traj_radiale: NDArray, theta: float = 0) -> NDArray:
    """
    Apply a rotation to a radial trajectory.
    
    Parameters
    ----------
    traj_radiale: NDArray 
        A 2D array representing the radial trajectory with shape (2, nb_points).
    theta: float, optional
        The rotation angle in degrees. Default is 0. The angle is converted to radians.

    Returns
    -------
    new_traj: NDArray
        The rotated radial trajectory with the same shape as the input `traj_radiale`.
    """
    theta = theta*2*np.pi/360
    
    c, s = np.cos(theta ), np.sin(theta )
    R = np.array(((c, -s), (s, c)))

    new_traj= traj_radiale@R

    return new_traj


def radial_sigpy(coord_shape: Tuple[int, int, int],
                 img_shape: Tuple[int, ...],
                 golden: bool = True,
                 half_spoke: bool = False,
                 dtype: type = float
                 ) -> NDArray:
    """
    Generate radial k-space coordinates for MRI, adapted from SIGPY: 
    https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.radial.html 

    Parameters
-   ---------
    coord_shape : tuple of int
        coordinates of shape [ntr, nro, ndim],
        where ntr is the number of TRs, nro is the number of readout,
        and ndim is the number of dimensions.
    img_shape : tuple of ints
        image shape.
    half_spoke : bool
        If True, generates a half-spoke radial trajectory/
    golden : bool
        If True, uses golden-angle ordering for the spoke directions
    dtype : np.dtype
        Data type of the output 

    Returns
    -------
    NDArray: 
        radial coordinates.

    References
    ----------
        1. An Optimal Radial Profile Order Based on the Golden
        Ratio for Time-Resolved MRI
        Stefanie Winkelmann, Tobias Schaeffter, Thomas Koehler,
        Holger Eggers, and Olaf Doessel. TMI 2007.
        2. Temporal stability of adaptive 3D radial MRI using
        multidimensional golden means
        Rachel W. Chan, Elizabeth A. Ramsay, Charles H. Cunningham,
        and Donald B. Plewes. MRM 2009.
    """
    if len(img_shape) != coord_shape[-1]:
        raise ValueError(
            "coord_shape[-1] must match len(img_shape), "
            "got {} and {}".format(coord_shape[-1], len(img_shape))
        )

    ntr, nro, ndim = coord_shape
    if ndim == 2:
        
        if golden:
            if half_spoke is True : 
                phi = np.pi * (3 - 5**0.5)
            else:
                phi = 111.25 * 2 * np.pi / 360

        else:
            phi = 2 * np.pi / ntr

        if half_spoke is True:
            n, r = np.mgrid[:ntr, : 0.5 : 0.5 / (nro)]
        else:
            n, r = np.mgrid[:ntr, -0.5: 0.5 : 0.5 / (nro/2)]
        theta = n * phi
        coord = np.zeros((ntr, nro, 2))
        coord[:, :, -1] = r * np.cos(theta)
        coord[:, :, -2] = r * np.sin(theta)

    else:
        raise ValueError("coord_shape[-1] must be 2 or 3, got {}".format(ndim))

    return (coord * img_shape[-ndim:]).astype(dtype)


def process_nufft(pulse: int,
                  image: NDArray,
                  radial_trajectory: NDArray,
                  radial_rotation: bool,
                  Nd: Tuple[int, int],
                  Kd: Tuple[int, int],
                  Jd: Tuple[int, int]
                  ) -> Tuple[int, NDArray, NDArray]:
    '''
    Processes one pulse for NUFFT computation and returns the corresponding
    k-space data and trajectory.
    
    Parameters
    ----------
    pulse : int
        Index of the current pulse to process.
    image : NDArray
        Image data array of shape (n_coils, n_pulses, 1, image_shape, image_shape).
    radial_trajectory : NDArray
        2D array of shape (2, nb_points), representing the radial k-space trajectory.
    radial_rotation : bool
        If True, applies angular rotation to the trajectory at each TR (pulse).
    Nd : tuple of int
        Image dimensions for NUFFT, typically (image_shape, image_shape).
    Kd : tuple of int
        Oversampled k-space grid size, typically (2×image_shape, 2×image_shape).
    Jd : tuple of int
        Size of the interpolation kernel, e.g., (6, 6).

    Returns
    -------
    pulse, k_space_nufft, conc_trajectory : Tuple[int, NDArray, NDArray]
        A tuple containing:
        - the pulse index (int),
        - the NUFFT-encoded k-space data for that pulse (NDArray),
        - the trajectory used for that pulse (NDArray).
    '''
    if radial_rotation is True :
        theta = pulse * 111.25
    else:
        theta = 0
    
    conc_trajectory = generate_radiale_traj(radial_trajectory.T,theta)
    
    om = conc_trajectory
    
    k_space_nufft = np.zeros((31, om.shape[0]), complex)

    obj_nufft = pynufft.NUFFT()
    obj_nufft.plan(om, Nd, Kd, Jd)

    for coil in range (31):
        k_space_nufft[coil,:] = obj_nufft.forward(image[coil, pulse, 0, :, :])
  
    return pulse, k_space_nufft, conc_trajectory


def radial_reconstruction(image: Optional[NDArray] = None,
                          k_space: Optional[NDArray] = None,
                          trajectory: Optional[NDArray] = None,
                          sensitivity_map: Optional[NDArray] = None,
                          n_iters: Optional[int] = None,
                          nspoke: Optional[int] = None,
                          rotationGA_spoke: bool = False,
                          radial_rotation: bool = True,
                          return_trajectory: bool = False
                          ) -> Union[NDArray, Tuple[NDArray, NDArray]]:    
    '''   
    Perform radial MRI reconstruction using CG-SENSE, from either k-space or image data.
    
    Parameters
    ----------
    image : NDArray, optional
        Input image data of shape (n_coils, n_pulses, 1, image_shape, image_shape).
    k_space : NDArray, optional
        Input k-space data of shape (n_coils, n_pulses, n_spokes * trajectory.shape[1]).
    trajectory : NDArray, optional
        The radial trajectory in k-space with shape (2, nb_points). If None, a trajectory 
        is generated using the provided parameters.
    sensitivity_map : NDArray, optional
        The sensitivity map for each coil with shape (n_coils, image_shape, image_shape). 
        Default is None.
    n_iters : int, optional
        Number of iterations for the conjugate gradient reconstruction algorithm.
    nspoke : int, optional
        Number of radial spokes to generate
    rotationGA_spoke : bool, optional
        Whether to use golden angle to define a multi-spoke trajectory. Default is False.
    radial_rotation : bool, optional
        Whether to apply radial rotation between TR to the trajectory. Default is True.  
    return_trajectory : bool, optional
        Whether to return the generated trajectory along with the reconstructed image. Default is False.

    Returns
    -------
    cg_sense or full_trajectory, cg_sense : NDArray or Tuple[NDArray, NDArray]
        - Reconstructed image of shape (n_pulses, image_shape, image_shape).
        - If `return_trajectory=True`, also returns the trajectory used
    '''
    if image is None and k_space is None :
        raise ValueError('Please provide either an image or a k-space')
    if image is not None and k_space is not None :
        raise ValueError('Please provide either an image or a k-space')
    
    if sensitivity_map is None :
        warnings.warn('No sensitivity map provided, the reconstruction will be done without it')
    else:
        sensitivity_map= tf.convert_to_tensor(sensitivity_map) 
    
    n_pulses=image.shape[1]
    n_coils=image.shape[0]
    image_shape = image.shape[-1]

    if trajectory is None :
        trajectory = radial_sigpy((nspoke,image_shape,2), (image_shape, image_shape), golden=rotationGA_spoke, half_spoke = False)
        trajectory = trajectory*np.pi / np.max(trajectory)
        trajectory = np.transpose(trajectory, (2,1,0)).reshape(2,-1)

    if image is not None :
            
        if image.ndim != 5 :
            raise ValueError('The image must have 5 dimensions')
        
        k_space_nufft_sum =  np.zeros((n_pulses, n_coils, trajectory.shape[1]), complex) 
        full_trajectory = []
        
        #The function is parallelized on each pulse
        results = Parallel(n_jobs=-1)(delayed(process_nufft)(pulse,image, trajectory, nspoke, radial_rotation, Nd = (image_shape,image_shape), Kd = (image_shape*2,image_shape*2), Jd=(6,6))for pulse in range(n_pulses))  
        
        #The results are stored in k_space_nufft_sum for all pulses and the associated trajectories are stored in tot_trajectory.
        for pulse, k_space_nufft, conc_trajectory in results:
            k_space_nufft_sum[pulse, :, :] = k_space_nufft
            full_trajectory.append(conc_trajectory)

    elif k_space is not None :
        if k_space.ndim != 3 :
            raise ValueError('The k-space must have 3 dimensions')

        k_space_nufft_sum = np.transpose(k_space, (1, 0, 2))
        full_trajectory = []

        for pulse in range (n_pulses):
            theta = (pulse * 111.25) % 360 
            conc_trajectory = generate_radiale_traj(trajectory,theta)
            full_trajectory.append(conc_trajectory)
        
    traj_tf = tf.convert_to_tensor( np.array(full_trajectory))
    traj_tf = tf.cast(traj_tf, tf.float32)
    k_space_tf = tf.convert_to_tensor(k_space_nufft_sum, dtype = tf.complex64)
    
    cg_sense = reconstruct_lstsq(k_space_tf, image_shape=[image_shape, image_shape], trajectory=traj_tf, sensitivities=sensitivity_map, optimizer='cg', optimizer_kwargs={'max_iterations':n_iters})

    if return_trajectory :
        return full_trajectory, cg_sense
    else :
        return  cg_sense