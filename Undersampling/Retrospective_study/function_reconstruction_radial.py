import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
from CG_SENSE.cg_sense_new_version import reconstruct_lstsq
import pynufft
import warnings


def generate_radiale_traj (traj_radiale, theta=0) :
    """
    Apply a rotation to a radial tr&ajectory.
    
    Args:
        traj_radiale (numpy.ndarray): A 2D array representing the radial trajectory with shape (2, nb_points).
        theta (float, optional): The rotation angle in degrees. Default is 0. The angle is converted to radians.
    
    Returns:
        numpy.ndarray: The rotated radial trajectory with the same shape as the input `traj_radiale`.
    """
    theta = theta*2*np.pi/360
    
    c, s = np.cos(theta ), np.sin(theta )
    R = np.array(((c, -s), (s, c)))

    new_matrix = traj_radiale@R

    return new_matrix

def radial_sigpy(coord_shape, img_shape, golden=True, half_spoke = False, dtype=float):
    """
    INSPIRED FROM SIGPY : Generate radial trajectory.

    Args:
        coord_shape (tuple of ints): coordinates of shape [ntr, nro, ndim],
            where ntr is the number of TRs, nro is the number of readout,
            and ndim is the number of dimensions.
        img_shape (tuple of ints): image shape.
        half_spoke (bool): if True, the radial trajectory is half-spoke.
        golden (bool): golden angle ordering.
        dtype (Dtype): data type.

    Returns:
        array: radial coordinates.

    References:
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


def process_nufft(pulse, image, radial_trajectory, radial_rotation, Nd, Kd, Jd):
    '''
    Process each pulse for NUFFT and return the k-space data and trajectory.
    
    Args:
        pulse (int): The pulse index.
        image (numpy.ndarray): The image data with shape (n_coils, n_pulses, 1, image_shape, image_shape).
        radial_trajectory (numpy.ndarray): The radial trajectory in k-space with shape (2, nb_points).
        radial_rotation (bool): Whether to apply radial rotation between TR to the trajectory.
        Nd (tuple): The dimensions of the image (image_shape, image_shape).
        Kd (tuple): The dimensions of the k-space (image_shape*2, image_shape*2).
        Jd (tuple): The dimensions of the kernel (6, 6).

    Returns:
        tuple: The pulse index, the k-space data for the given pulse, and the generated trajectory for that pulse.
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



def radial_reconstruction(image=None, k_space=None, trajectory=None,sensitivity_map=None, n_iters=None, nspoke=None, rotationGA_spoke = False, radial_rotation=True, return_trajectory=False):
    '''   
    Perform radial MRI reconstruction from either k-space data or an image using CG-SENSE.
    
    Args:
        image (numpy.ndarray, optional): The image data with shape (n_coils, n_pulses, 1, image_shape, image_shape). 
        k_space (numpy.ndarray, optional): The k-space data with shape (n_coils, n_pulses, nspokes * trajectory.shape[1]).
        trajectory (numpy.ndarray, optional): The radial trajectory in k-space with shape (2, nb_points). If None, a trajectory is generated using the provided parameters.
        sensitivity_map (numpy.ndarray, optional): The sensitivity map for each coil with shape (n_coils, image_shape, image_shape). Default is None.
        n_iters (int, optional): The number of iterations for the reconstruction. Default is None.
        f_sampling (float, optional): The undersampling factor in the frequency encoding direction. Default is None.
        R (float, optional): The undersampling factor in the spiral trajectory. Default is None.
        nspoke (int, optional): The number of spoke. Default is None.
        rotationGA_spoke (bool, optional): Whether to use golden angle to define a multi-spoke trajectory. Default is False.
        radial_rotation (bool, optional): Whether to apply radial rotation between TR to the trajectory. Default is True.  
        return_trajectory (bool, optional): Whether to return the generated trajectory along with the reconstructed image. Default is False.

    Returns:
        numpy.ndarray: The reconstructed image for each pulse. 
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