import numpy as np
from joblib import Parallel, delayed
import tensorflow as tf
import sys
import os
import warnings
import pynufft
import sigpy.mri

sys.path.append(os.path.join(os.getcwd(), os.pardir, os.pardir))

from CG_SENSE.cg_sense_new_version import reconstruct_lstsq


def generate_spirale_traj (n_spirale, traj_spirale, theta=0) :
    """
    Apply a rotation to a radial trajectory.

    Args:
        traj_radiale (numpy.ndarray): A 2D array representing the radial trajectory with shape (2, nb_points).
        theta (float, optional): The rotation angle in degrees. Default is 0. The angle is converted to radians.

    Returns:
        numpy.ndarray: The rotated radial trajectory with the same shape as the input `traj_radiale`.
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


def spiral_sigpy(fov, N, f_sampling, R, ninterleaves, alpha, gm, sm, gamma=2.678e8):
    """
    FROM SIGPY : Generate variable density spiral trajectory.

    Args:
        fov (float): field of view in meters.
        N (int): effective matrix shape.
        f_sampling (float): undersampling factor in freq encoding direction.
        R (float): undersampling factor.
        ninterleaves (int): number of spiral interleaves
        alpha (float): variable density factor
        gm (float): maximum gradient amplitude (T/m)
        sm (float): maximum slew rate (T/m/s)
        gamma (float): gyromagnetic ratio in rad/T/s

    Returns:
        array: spiral coordinates.

    References:
        Dong-hyun Kim, Elfar Adalsteinsson, and Daniel M. Spielman.
        'Simple Analytic Variable Density Spiral Design.' MRM 2003.

    """
    return sigpy.mri.spiral(fov=fov,N=N,f_sampling=f_sampling,R=R,ninterleaves=ninterleaves,alpha=alpha,gm=gm,sm=sm) 
          

def process_nufft(pulse, trajectory, ninterleaves, image, spiral_rotation, Nd, Kd, Jd):
    """
    Perform Non-Uniform Fourier Transform (NUFFT) to obtain a k-space from an image 
    according to the non-cartesian trajectory.

    Args:
        pulse (int): The pulse index in the sequence.
        trajectory (numpy.ndarray): The spiral trajectory with shape (2, n_points).
        ninterleaves (int): The number of spiral interleaves (arms).
        image (numpy.ndarray): The multi-coil image data with shape (n_coils, n_pulses, 1, image_shape, image_shape).
        Nd (tuple): The size of the image grid in the frequency domain (k-space).
        Kd (tuple): The size of the k-space grid.
        Jd (tuple): The number of points used for the NUFFT interpolation in each dimension.

    Returns:
        tuple: The pulse index, the k-space data for the given pulse, and the generated trajectory for that pulse.
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



def spiral_reconstruction(image=None, k_space=None, trajectory=None, full_trajectory = False, sensitivity_map=None, n_iters=None, f_sampling=None, R=None, ninterleaves=None, alpha=None, spiral_rotation=True, return_trajectory=False):
    '''  
    Perform spiral MRI reconstruction from either k-space data or an image using CG-SENSE.
    
    Args:
        image (numpy.ndarray, optional): The image data with shape (n_coils, n_pulses, 1, image_shape, image_shape). 
        k_space (numpy.ndarray, optional): The k-space data with shape (n_coils, n_pulses, ninterleaves * trajectory.shape[1]).
        trajectory (numpy.ndarray, optional): The spiral trajectory in k-space with shape (2, nb_points). If None, a trajectory is generated using the provided parameters.
        sensitivity_map (numpy.ndarray, optional): The sensitivity map for each coil with shape (n_coils, image_shape, image_shape). Default is None.
        n_iters (int, optional): The number of iterations for the reconstruction. Default is None.
        f_sampling (float, optional): The undersampling factor in the frequency encoding direction. Default is None.
        R (float, optional): The undersampling factor in the spiral trajectory. Default is None.
        ninterleaves (int, optional): The number of spiral interleaves (arms). Default is None.
        alpha (float, optional): The variable density factor. Default is None.
        return_trajectory (bool, optional): Whether to return the generated trajectory along with the reconstructed image. Default is False.

    Returns:
        numpy.ndarray: The reconstructed image for each pulse. 
    '''
    
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