import tensorflow as tf
import sys 
import os

sys.path.append(os.path.join(os.getcwd(), os.pardir))

import CG_SENSE.convex_ops as convex_ops
import CG_SENSE.fft as fft
import CG_SENSE.linalg_ops  as linalg_ops
import CG_SENSE.utils as utils

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def reconstruct_lstsq(kspace,
                      image_shape,
                      extra_shape=None,
                      mask=None,
                      trajectory=None,
                      density=None,
                      sensitivities=None,
                      phase=None,
                      sens_norm=True,
                      dynamic_domain=None,
                      regularizer=None,
                      optimizer=None,
                      optimizer_kwargs=None,
                      filter_corners=False,
                      return_optimizer_state=False,
                      toeplitz_nufft=False):
  r"""Reconstructs an MR image using a least-squares formulation.

  This is an iterative reconstruction method which formulates the image
  reconstruction problem as follows:

  .. math::
    \hat{x} = {\mathop{\mathrm{argmin}}_x} \left (\left\| Ax - y \right\|_2^2 + g(x) \right )

  where :math:`A` is the MRI `LinearOperator`, :math:`x` is the solution, `y` is
  the measured *k*-space data, and :math:`g(x)` is an optional `ConvexFunction`
  used for regularization.

  This operator supports Cartesian and non-Cartesian *k*-space data.

  This operator supports linear and non-linear reconstruction, depending on the
  selected regularizer. The MRI operator is constructed internally and does not
  need to be provided.

  This operator supports batched inputs. All batch shapes should be
  broadcastable with each other.

  Args:
    kspace: A `Tensor`. The *k*-space samples. Must have type `complex64` or
      `complex128`. `kspace` can be either Cartesian or non-Cartesian. A
      Cartesian `kspace` must have shape
      `[..., *extra_shape, num_coils, *image_shape]`, where `...` are batch
      dimensions. A non-Cartesian `kspace` must have shape
      `[..., *extra_shape, num_coils, num_samples]`.
    image_shape: A `TensorShape` or a list of `ints`. Must have length 2 or 3.
      The shape of the reconstructed image[s].
    extra_shape: An optional `TensorShape` or list of `ints`. Additional
      dimensions that should be included within the solution domain. Note
      that `extra_shape` is not needed to reconstruct independent batches of
      images. However, it should be provided when performing a reconstruction
      that operates along non-spatial dimensions, e.g. for temporal
      regularization. Defaults to `[]`.
    mask: An optional `Tensor` of type `bool`. The sampling mask. Must have
      shape `[..., image_shape]`. `mask` should be passed for reconstruction
      from undersampled Cartesian *k*-space. For each point, `mask` should be
      `True` if the corresponding *k*-space sample was measured and `False`
      otherwise.
    trajectory: An optional `Tensor` of type `float32` or `float64`. Must have
      shape `[..., num_samples, rank]`. `trajectory` should be passed for
      reconstruction from non-Cartesian *k*-space.
    density: An optional `Tensor` of type `float32` or `float64`. The sampling
      densities. Must have shape `[..., num_samples]`. This input is only
      relevant for non-Cartesian MRI reconstruction. If passed, the MRI linear
      operator will include sampling density compensation. If `None`, the MRI
      operator will not perform sampling density compensation.
    sensitivities: An optional `Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape
      `[..., num_coils, *image_shape]`. If provided, a multi-coil parallel
      imaging reconstruction will be performed.
    phase: An optional `Tensor` of type `float32` or `float64`. Must have shape
      `[..., *image_shape]`. A phase estimate for the reconstructed image. If
      provided, a phase-constrained reconstruction will be performed. This
      improves the conditioning of the reconstruction problem in applications
      where there is no interest in the phase data. However, artefacts may
      appear if an inaccurate phase estimate is passed.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.
    dynamic_domain: A `str`. The domain of the dynamic dimension, if present.
      Must be one of `'time'` or `'frequency'`. May only be provided together
      with a non-scalar `extra_shape`. The dynamic dimension is the last
      dimension of `extra_shape`. The `'time'` mode (default) should be
      used for regular dynamic reconstruction. The `'frequency'` mode should be
      used for reconstruction in x-f space.
    regularizer: A `ConvexFunction`. The regularization term added to
      least-squares objective.
    optimizer: A `str`. One of `'cg'` (conjugate gradient), `'admm'`
      (alternating direction method of multipliers) of `'lbfgs'`
      (limited-memory Broyden-Fletcher-Goldfarb-Shanno). If `None`, the
      optimizer is selected heuristically depending on other inputs. Note that
      this heuristic may change in the future, so specify an optimizer if you
      wish to ensure it will always be used in future versions. Not all
      optimizers are compatible with all configurations.
    optimizer_kwargs: An optional `dict`. Additional arguments to pass to the
      optimizer.
    filter_corners: A `boolean`. Whether to filter out the *k*-space corners in
      reconstructed image. This may be done for trajectories with a circular
      *k*-space coverage. Defaults to `False`.
    return_optimizer_state: A `boolean`. If `True`, returns the optimizer
      state along with the reconstructed image.
    toeplitz_nufft: A `boolean`. If `True`, uses the Toeplitz approach [5]
      to compute :math:`F^H F x`, where :math:`F` is the non-uniform Fourier
      operator. If `False`, the same operation is performed using the standard
      NUFFT operation. The Toeplitz approach might be faster than the direct
      approach but is slightly less accurate. This argument is only relevant
      for non-Cartesian reconstruction and will be ignored for Cartesian
      problems.

  Returns:
    A `Tensor`. The reconstructed image. Has the same type as `kspace` and
    shape `[..., *extra_shape, *image_shape]`, where `...` is the broadcasted
    batch shape of all inputs.

    If `return_optimizer_state` is `True`, returns a tuple containing the
    reconstructed image and the optimizer state.

  Raises:
    ValueError: If passed incompatible inputs.

  Notes:
    Reconstructs an image by formulating a (possibly regularized) least squares
    problem, which is solved iteratively. Since the problem may be ill-posed,
    different types of regularizers may be used to incorporate prior knowledge.
    Depending on the regularizer, the optimization problem may be linear or
    nonlinear. For sparsity-based regularizers, this is also called a compressed
    sensing reconstruction. This is a powerful operator which can often produce
    high-quality images even from highly undersampled *k*-space data. However,
    it may be time-consuming, depending on the characteristics of the problem.

  References:
    .. [1] Pruessmann, K.P., Weiger, M., Börnert, P. and Boesiger, P. (2001),
      Advances in sensitivity encoding with arbitrary k-space trajectories.
      Magn. Reson. Med., 46: 638-651. https://doi.org/10.1002/mrm.1241

    .. [2] Block, K.T., Uecker, M. and Frahm, J. (2007), Undersampled radial MRI
      with multiple coils. Iterative image reconstruction using a total
      variation constraint. Magn. Reson. Med., 57: 1086-1098.
      https://doi.org/10.1002/mrm.21236

    .. [3] Feng, L., Grimm, R., Block, K.T., Chandarana, H., Kim, S., Xu, J.,
      Axel, L., Sodickson, D.K. and Otazo, R. (2014), Golden-angle radial sparse
      parallel MRI: Combination of compressed sensing, parallel imaging, and
      golden-angle radial sampling for fast and flexible dynamic volumetric MRI.
      Magn. Reson. Med., 72: 707-717. https://doi.org/10.1002/mrm.24980

    .. [4] Tsao, J., Boesiger, P., & Pruessmann, K. P. (2003). k-t BLAST and
      k-t SENSE: dynamic MRI with high frame rate exploiting spatiotemporal
      correlations. Magnetic Resonance in Medicine: An Official Journal of the
      International Society for Magnetic Resonance in Medicine, 50(5),
      1031-1042.

    .. [5] Fessler, J. A., Lee, S., Olafsson, V. T., Shi, H. R., & Noll, D. C.
      (2005). Toeplitz-based iterative image reconstruction for MRI with
      correction for magnetic field inhomogeneity. IEEE Transactions on Signal
      Processing, 53(9), 3393-3402.
  """  # pylint: disable=line-too-long
  device =  '/GPU:0' if tf.config.list_physical_devices('GPU') else'/CPU:0'
  with tf.device(device):
    # Choose a default optimizer.
    if optimizer is None:
      if regularizer is None or isinstance(regularizer,
                                          convex_ops.ConvexFunctionTikhonov):
        optimizer = 'cg'
      else:
        optimizer = 'admm'
    # Check optimizer.
      optimizer = utils.validate_enum(
          optimizer, {'cg', 'admm', 'lbfgs'}, name='optimizer')
    optimizer_kwargs = optimizer_kwargs or {}

    # We don't do a lot of input checking here, since it will be done by the
    # operator.
    kspace = tf.convert_to_tensor(kspace)

    # Create the linear operator.
    operator = linalg_ops.LinearOperatorMRI(image_shape,
                                            extra_shape=extra_shape,
                                            mask=mask,
                                            trajectory=trajectory,
                                            density=density,
                                            sensitivities=sensitivities,
                                            phase=phase,
                                            fft_norm='ortho',
                                            sens_norm=sens_norm,
                                            dynamic_domain=dynamic_domain)
    rank = operator.rank

    # If using Toeplitz NUFFT, we need to use the specialized Gram MRI operator.
    if toeplitz_nufft and operator.is_non_cartesian:
      gram_operator = linalg_ops.LinearOperatorGramMRI(
          image_shape,
          extra_shape=extra_shape,
          mask=mask,
          trajectory=trajectory,
          density=density,
          sensitivities=sensitivities,
          phase=phase,
          fft_norm='ortho',
          sens_norm=sens_norm,
          dynamic_domain=dynamic_domain,
          toeplitz_nufft=toeplitz_nufft)
    else:
      # No Toeplitz NUFFT. In this case don't bother defining the Gram operator.
      gram_operator = None

    # Apply density compensation, if provided.
    if density is not None:
      kspace *= operator._dens_weights_sqrt  # pylint: disable=protected-access

    initial_image = operator.H.transform(kspace)

    # Optimizer-specific logic.
    if optimizer == 'cg':
      if regularizer is not None:
        if not isinstance(regularizer, convex_ops.ConvexFunctionTikhonov):
          raise ValueError(
              f"Regularizer {regularizer.name} is incompatible with "
              f"CG optimizer.")
        reg_parameter = regularizer.function.scale
        reg_operator = regularizer.transform
        reg_prior = regularizer.prior
      else:
        reg_parameter = None
        reg_operator = None
        reg_prior = None

      operator_gm = linalg_ops.LinearOperatorGramMatrix(
          operator, reg_parameter=reg_parameter, reg_operator=reg_operator,
          gram_operator=gram_operator)
      rhs = initial_image
      # Update the rhs with the a priori estimate, if provided.
      if reg_prior is not None:
        if reg_operator is not None:
          reg_prior = reg_operator.transform(
              reg_operator.transform(reg_prior), adjoint=True)
        rhs += tf.cast(reg_parameter, reg_prior.dtype) * reg_prior
      # Solve the (maybe regularized) linear system.
      result = linalg_ops.conjugate_gradient(operator_gm, rhs, **optimizer_kwargs)
      image = result.x

    elif optimizer == 'admm':
      if regularizer is None:
        raise ValueError("optimizer 'admm' requires a regularizer")
      # Create the least-squares objective.
      function_f = convex_ops.ConvexFunctionLeastSquares(
          operator, kspace, gram_operator=gram_operator)
      # Configure ADMM formulation depending on regularizer.
      if isinstance(regularizer,
                    convex_ops.ConvexFunctionLinearOperatorComposition):
        function_g = regularizer.function
        operator_a = regularizer.operator
      else:
        function_g = regularizer
        operator_a = None
      # Run ADMM minimization.
      result = linalg_ops.admm_minimize(function_f, function_g,
                                          operator_a=operator_a,
                                          **optimizer_kwargs)
      image = operator.expand_domain_dimension(result.f_primal_variable)

    elif optimizer == 'lbfgs':
      # Flatten k-space and initial estimate.
      initial_image = operator.flatten_domain_shape(initial_image)
      y = operator.flatten_range_shape(kspace)

      # Currently L-BFGS implementation only supports real numbers, so reinterpret
      # complex image as real (C^N -> R^2*N).
      initial_image = utils.view_as_real(initial_image, stacked=False)

      # Define the objective function and its gradient.
      @utils.make_val_and_grad_fn
      def _objective(x):
        # Reinterpret real input as complex.
        x = utils.view_as_complex(x, stacked=False)
        # Compute objective.
        obj = tf.math.abs(tf.norm(y - operator.matvec(x), ord=2))
        print('Regularization start')
        print(obj)
        print(obj.shape)

        if regularizer is not None:
          obj += regularizer(x)
        print('Regularization has been done')
        return obj

      # Do minimization.
      result = linalg_ops.lbfgs_minimize(_objective, initial_image,
                                            **optimizer_kwargs)

      # Reinterpret real result as complex and reshape image.
      image = operator.expand_domain_dimension(
          utils.view_as_complex(result.position, stacked=False))

    else:
      raise ValueError(f"Unknown optimizer: {optimizer}")

    # Apply temporal Fourier operator, if necessary.
    if operator.is_dynamic and operator.dynamic_domain == 'frequency':
      image = fft.ifftn(image, axes=[operator.dynamic_axis],
                            norm='ortho', shift=True)

    # Apply intensity correction, if requested.
    if operator.is_multicoil and sens_norm:
      sens_weights_sqrt = tf.math.reciprocal_no_nan(
          tf.norm(sensitivities, axis=-(rank + 1), keepdims=False))
      image *= sens_weights_sqrt

    # If necessary, filter the image to remove k-space corners. This can be
    # done if the trajectory has circular coverage and does not cover the k-space
    # corners. If the user has not specified whether to apply the filter, we do it
    # only for non-Cartesian trajectories, under the assumption that non-Cartesian
    # trajectories are likely to have circular coverage of k-space while Cartesian
    # trajectories are likely to have rectangular coverage.
    if filter_corners is None:
      is_probably_circular = operator.is_non_cartesian
      filter_corners = is_probably_circular
    if filter_corners:
      fft_axes = list(range(-rank, 0))  # pylint: disable=invalid-unary-operand-type
      kspace = fft.fftn(image, axes=fft_axes, norm='ortho', shift=True)
      kspace = utils.filter_kspace(kspace, filter_fn='atanfilt',
                                        filter_rank=rank)
      image = fft.ifftn(kspace, axes=fft_axes, norm='ortho', shift=True)

    if return_optimizer_state:
      return image, result

    return image
