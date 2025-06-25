import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import sys
import os
import warnings
import pynufft
import sigpy.mri
from numpy.typing import NDArray
from typing import Tuple, Optional, Union

sys.path.append(os.path.join(os.getcwd(), os.pardir, os.pardir))

from CG_SENSE.cg_sense_new_version import reconstruct_lstsq


def generate_spirale_traj(n_spirale: int,
                          traj_spirale: NDArray,
                          theta: float = 0
                          ) -> NDArray:
    """
    Create an n-shot spiral trajectory by applying rotations (2pi/n_spirale) to an initial 
    trajectory rotated by a theta angle.

    Parameters
    ----------
    n_spirale : int
        Number of shots.
    traj_spirale : NDArray
        A 2D array representing the spiral trajectory with shape (2, nb_points).
    theta : float, optional
        Global rotation angle (in degrees) applied to the first shot. Default is 0°.
        The angle is internally converted to radians.

    Returns
    -------
    conc_trajectory : NDArray
        A 2D array of shape (2, nb_points * n_spirale) representing the combined spiral trajectory.
    """
    theta = theta*2*np.pi/360
   
    for i in range(n_spirale) : 
        
        if i ==0:
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))

            new_matrix = R@traj_spirale
            conc_trajectory = new_matrix
        
        else:
            angle = theta + i*2*np.pi/n_spirale
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, -s), (s, c)))

            new_matrix = R@traj_spirale
            conc_trajectory = np.concatenate((conc_trajectory, new_matrix), axis=1)
    
    return conc_trajectory


def spiral_sigpy(fov: float,
                 N: int,
                 f_sampling: float,
                 R: float,
                 ninterleaves: int,
                 alpha: float,
                 gm: float,
                 sm: float,
                 gamma: float = 2.678e8) -> NDArray:
    """
    FROM SIGPY : https://sigpy.readthedocs.io/en/latest/generated/sigpy.mri.spiral.html#sigpy.mri.spiral
    Generate a variable density spiral trajectory.

    Parameters
    ----------
    fov : float
        Field of view in meters.
    N : int
        Effective matrix size.
    f_sampling : float
        Undersampling factor in the frequency encoding direction.
    R : float
        Undersampling factor.
    ninterleaves : int
        Number of spiral interleaves.
    alpha : float 
        variable density factor.
    gm : float
        Maximum gradient amplitude(T/m)
    sm : float 
        Maximum slew rate (T/m/s)
    gamma : float, optional
        gyromagnetic ratio in rad/T/s

    Returns
    -------
    NDArray: 
        spiral coordinates.

    References
    ----------
        Dong-hyun Kim, Elfar Adalsteinsson, and Daniel M. Spielman.
        'Simple Analytic Variable Density Spiral Design.' MRM 2003.
    """
    res = fov / N

    lam = 0.5 / res  # in m**(-1)
    n = 1 / (1 - (1 - ninterleaves * R / fov / lam) ** (1 / alpha))

    w = 2 * np.pi * n
    Tea = lam * w / gamma / gm / (alpha + 1)  # in s
    Tes = np.sqrt(lam * w**2 / sm / gamma) / (alpha / 2 + 1)  # in s
    Ts2a = (
        Tes ** ((alpha + 1) / (alpha / 2 + 1))
        * (alpha / 2 + 1)
        / Tea
        / (alpha + 1)
    ) ** (
        1 + 2 / alpha
    )  # in s

    if Ts2a < Tes:
        tautrans = (Ts2a / Tes) ** (1 / (alpha / 2 + 1))

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= Ts2a) + (np.abs((t - Ts2a)) / Tea + tautrans ** (alpha + 1)) ** (1 / (alpha + 1)) * (t > Ts2a) * (t <= Tea) * (Tes >= Ts2a)

        Tend = Tea
    else:

        def tau(t):
            return (t / Tes) ** (1 / (alpha / 2 + 1)) * (0 <= t) * (t <= Tes)

        Tend = Tes

    def k(t):
        return lam * tau(t) ** alpha * np.exp(w * tau(t) * 1j)

    dt = Tea * 1e-4  # in s

    Dt = dt * f_sampling / fov / abs(k(Tea) - k(Tea - dt))  # in s

    t = np.linspace(0, Tend, int(Tend / Dt))

    kt = k(t)  # in rad

    # generating cloned interleaves
    k = kt
    for i in range(1, ninterleaves):
        k = np.hstack((k, kt[0:] * np.exp(2 * np.pi * 1j * i / ninterleaves)))

    k = np.stack((np.real(k), np.imag(k)), axis=1)
    return k
          

def process_nufft(pulse: int,
                  trajectory: NDArray,
                  ninterleaves: int,
                  image: NDArray,
                  spiral_rotation: bool,
                  Nd: Tuple[int, int],
                  Kd: Tuple[int, int],
                  Jd: Tuple[int, int]
                  ) -> Tuple[int, NDArray, NDArray]:
    """
    Perform Non-Uniform Fourier Transform (NUFFT) to obtain a k-space from an image 
    according to the non-cartesian trajectory.

    Parameters
    ----------
    pulse : int
        The pulse index in the sequence.
    trajectory : NDArray
        The spiral trajectory of shape (2, n_points).
    ninterleaves : int 
        The number of spiral interleaves (rotations of the base trajectory).
    image : NDArray 
        The multi-coil image data with shape (n_coils, n_pulses, 1, image_shape, image_shape).
    spiral_rotation : bool
        Whether to rotate the trajectory between pulses.
    Nd : tuple of int
        The size of the image grid, (image_shape, image_shape).
    Kd : tuple of int
        The size of the k-space grid, usually (2×image_shape, 2×image_shape).
    Jd : tuple of int
        Interpolation kernel size for NUFFT in each dimension.

    Returns
    -------
    pulse, k_space_nufft, conc_trajectory : tuple
        A tuple containing:
        - `pulse` (int): the pulse index,
        - `kspace_data` (NDArray): NUFFT-encoded k-space data for that pulse,
        - `rotated_trajectory` (NDArray): the trajectory used for that pulse (2, n_points).
    """
    if spiral_rotation is True :
        theta = (pulse * 137.51) % 360 # Increase the angle by 137.51° (the golden angle) between each pulse to induce spiral rotation.
    else:
        theta=0

    conc_trajectory = generate_spirale_traj(ninterleaves, trajectory,theta) #Generate the trajectory
    om = conc_trajectory.T
    
    k_space_nufft = np.zeros((31, om.shape[0]), complex)

    obj_nufft = pynufft.NUFFT()
    obj_nufft.plan(om, Nd, Kd, Jd)

    for coil in range (31):
        k_space_nufft[coil,:] = obj_nufft.forward(image[coil, pulse, 0, :, :]) #Obtain k_space associate to the trajectory and the image by nufft
    
    return pulse, k_space_nufft, conc_trajectory.T


def spiral_reconstruction(image: Optional[NDArray] = None,
                          k_space: Optional[NDArray] = None,
                          trajectory: Optional[NDArray] = None,
                          full_trajectory: bool = False,
                          sensitivity_map: Optional[NDArray] = None,
                          n_iters: Optional[int] = None,
                          f_sampling: Optional[float] = None,
                          R: Optional[float] = None,
                          ninterleaves: Optional[int] = None,
                          alpha: Optional[float] = None,
                          spiral_rotation: bool = True,
                          return_trajectory: bool = False
                          ) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    Perform spiral MRI reconstruction from either k-space data or an image using CG-SENSE.

    Parameters
    ----------
    image : NDArray, optional
        The image data with shape (n_coils, n_pulses, 1, image_shape, image_shape).
    k_space : NDArray, optional
        The k-space data with shape (n_coils, n_pulses, ninterleaves * trajectory.shape[1]).
    trajectory : NDArray, optional
        The spiral trajectory in k-space with shape (2, nb_points).
        If None, a trajectory is generated using the provided parameters.
    full_trajectory : bool, optional
        If True, the trajectory given is already the trajectory for multiple shots. Only when the k_space is given.
    sensitivity_map : NDArray, optional
        The sensitivity map for each coil with shape (n_coils, image_shape, image_shape). Default is None.
    n_iters : int, optional
        The number of iterations for the reconstruction. Default is None.
    f_sampling : float, optional
        The undersampling factor in the frequency encoding direction. Default is None
    R : float, optional
        The undersampling factor in the spiral trajectory. Default is None.
    ninterleaves : int, optional
        The number of spiral interleaves. Default is None.
    alpha : float, optional
        The variable density factor. Default is None.
    spiral_rotation : bool, optional
        Whether to apply a rotation between spiral interleaves across pulses. Default is True.
    return_trajectory : bool, optional
        Whether to return the generated trajectory along with the reconstructed image. Default is False.

    Returns
    -------
    cg_sense or full_trajectory, cg_sense : NDArray or Tuple[NDArray, NDArray]
        The reconstructed image for each pulse.
        If `return_trajectory` is True, also returns the trajectory used.
    """
    
    if image is None and k_space is None :
        raise ValueError('Please provide either an image or a k-space')
    if image is not None and k_space is not None :
        raise ValueError('Please provide either an image or a k-space')
    
    if trajectory is None and (f_sampling or R  or alpha) is None :
        raise ValueError('Please provide either a trajectory or the parameters to generate one')
    if trajectory is not None and (f_sampling or R  or alpha) is not None :
        raise ValueError('Please provide either a trajectory or the parameters to generate one')

    if image is not None :
        n_pulses=image.shape[1]
        n_coils=image.shape[0]
        image_shape = image.shape[-1]
        fov = image_shape * 0.001

    else :
        image_shape = sensitivity_map.shape[-1] 
        n_pulses = k_space.shape[1]
        n_coils = k_space.shape[0]

    if sensitivity_map is None :
        warnings.warn('No sensitivity map provided, the reconstruction will be done without it')
    else:
        sensitivity_map= tf.convert_to_tensor(sensitivity_map, dtype = tf.complex64) 
    
    if trajectory is None :
        trajectory = spiral_sigpy(fov=fov, N=image_shape, f_sampling=f_sampling, R=R, ninterleaves=ninterleaves, alpha=alpha, gm=0.004, sm=200)
        trajectory = trajectory*np.pi / np.max(trajectory)
        trajectory = trajectory.T
        ninterleaves = 1

    if image is not None :
            
        if image.ndim != 5 :
            raise ValueError('The image must have 5 dimensions')
        
        k_space_nufft_sum =  np.zeros((n_pulses, n_coils, ninterleaves* trajectory.shape[1]), complex) 
        
        full_trajectory = []
        
        #The function is parallelized on each pulse
        results = Parallel(n_jobs=-1)(delayed(process_nufft)(pulse, trajectory, ninterleaves, image, spiral_rotation, Nd = (image_shape,image_shape), Kd = (image_shape*2,image_shape*2), Jd=(6,6))for pulse in range(n_pulses))  
        
        #The results are stored in k_space_nufft_sum for all pulses and the associated trajectories are stored in tot_trajectory.
        for pulse, k_space_nufft, conc_trajectory in results:
            k_space_nufft_sum[pulse, :, :] = k_space_nufft
            full_trajectory.append(conc_trajectory)

    elif k_space is not None :
        if k_space.ndim != 3 :
            raise ValueError('The k-space must have 3 dimensions')

        k_space_nufft_sum = np.transpose(k_space, (1, 0, 2))

        if full_trajectory is False : 
            
            full_trajectory = []

            for pulse in range (n_pulses):
                theta = (pulse * 137.51) % 360 # Increase the angle by 137.51° (the golden angle) between each pulse to induce spiral rotation.
                conc_trajectory = generate_spirale_traj(ninterleaves, trajectory,theta) #Generate the trajectory
    
                full_trajectory.append(conc_trajectory.T)

        else :
            full_trajectory = trajectory        

    traj_tf = tf.convert_to_tensor( np.array(full_trajectory))
    traj_tf = tf.cast(traj_tf, tf.float32)

    k_space_tf = tf.convert_to_tensor(k_space_nufft_sum, dtype = tf.complex64)
    
    cg_sense = reconstruct_lstsq(k_space_tf, image_shape=[image_shape, image_shape], trajectory=traj_tf, sensitivities=sensitivity_map, optimizer='cg', optimizer_kwargs={'max_iterations':n_iters})

    if return_trajectory :
        return full_trajectory, cg_sense
    else :
        return  cg_sense