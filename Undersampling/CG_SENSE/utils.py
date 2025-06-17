import tensorflow as tf
import CG_SENSE.convex_ops as convex_ops
import functools
import tensorflow_probability as tfp
import numpy as np
from tensorflow_probability.python.internal.prefer_static import *


def _prefer_static(original_fn, static_fn, disable_spec_check=False):
  """Wraps original_fn, preferring to call static_fn when inputs are static."""
  original_spec = (
      tf_inspect.getfullargspec(original_fn)._replace(annotations={}))
  static_spec = tf_inspect.getfullargspec(static_fn)._replace(annotations={})
  if not disable_spec_check and original_spec != static_spec:
    raise ValueError(
        'Arg specs do not match: original={}, static={}, fn={}'.format(
            original_spec, static_spec, original_fn))
  @decorator.decorator
  def wrap(wrapped_fn, *args, **kwargs):
    """The actual wrapper."""
    del wrapped_fn
    flat_args = tf.nest.flatten([args, kwargs])
    # N.B.: This `get_static_value` is nontrivial even in Eager mode, because
    # Keras's symbolic Tensors can exist when executing eagerly, and their
    # static values can be `None`.
    flat_args_ = [tf.get_static_value(a) for a in flat_args]
    all_static = all(arg is None or arg_ is not None
                     for arg, arg_ in zip(flat_args, flat_args_))
    if all_static:
      [args_, kwargs_] = tf.nest.pack_sequence_as([args, kwargs], flat_args_)
      return static_fn(*args_, **kwargs_)
    return original_fn(*args, **kwargs)
  return wrap(original_fn)


concat = _prefer_static(tf.concat, nptf.concat)

def broadcast_shape(x_shape, y_shape):
  """Computes the shape of a broadcast.

  When both arguments are statically-known, the broadcasted shape will be
  computed statically and returned as a `TensorShape`.  Otherwise, a rank-1
  `Tensor` will be returned.

  Args:
    x_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.
    y_shape: A `TensorShape` or rank-1 integer `Tensor`.  The input `Tensor` is
      broadcast against this shape.

  Returns:
    shape: A `TensorShape` or rank-1 integer `Tensor` representing the
      broadcasted shape.
  """
  x_shape_static = tf.get_static_value(x_shape)
  y_shape_static = tf.get_static_value(y_shape)
  if (x_shape_static is None) or (y_shape_static is None):
    return tf.broadcast_dynamic_shape(x_shape, y_shape)

  return tf.broadcast_static_shape(
      tf.TensorShape(x_shape_static), tf.TensorShape(y_shape_static))

def convert_shape_to_tensor(shape, name=None):
  """Convert a static shape to a tensor."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = tf.dtypes.int32
  else:
    dtype = None
  return tf.convert_to_tensor(shape, dtype=dtype, name=name)

def validate_type(value, type_, name=None):
  """Validates that value is of the specified type.

  Args:
    value: The value to validate.
    type_: The requested type.
    name: The name of the argument being validated. This is only used to format
      error messages.

  Returns:
    A valid value of type `type_`.

  Raises:
    TypeError: If `value` does not have type `type_`.
  """
  if not isinstance(value, type_):
    raise TypeError(
      f"Argument `{name}` must be of type {type_}, "
      f"but received type: {type(value)}")
  return value

def validate_rank(value, name=None, accept_none=True):
  """Validates that `value` is a valid rank.

  Args:
    value: The value to check.
    name: The name of the parameter. Only used to format error messages.
    accept_none: If `True`, `None` is accepted as a valid value.

  Returns:
    The value.

  Raises:
    TypeError: If `value` has an invalid type.
    ValueError: If `value` is not a valid rank.
  """
  if value is None:
    if accept_none:
      return None
    raise ValueError(f'Argument `{name}` must be specified.')
  if not isinstance(value, int):
    raise TypeError(
        f'Argument `{name}` must be an integer, but got {value}.')
  if value < 0:
    raise ValueError(
        f'Argument `{name}` must be non-negative, but got {value}.')
  return value

def validate_list(value,
                  element_type=None,
                  length=None,
                  broadcast_scalars=True,
                  allow_tuples=True,
                  name=None):
  """Validates that value is a list with the specified characteristics.

  Args:
    value: The value to validate.
    element_type: A `type` or tuple of `type`s. The expected type for elements
      of the input list. Can be a tuple to allow more than one type. If `None`,
      the element type is not enforced.
    length: An `int`. The expected length of the list. If `None`, the length is
      not enforced.
    broadcast_scalars: A `boolean`. If `True`, scalar inputs are converted to
      lists of length `length`, if `length` is not `None`, or length 1
      otherwise. If `False`, an error is raised on scalar inputs.
    allow_tuples: A `boolean`. If `True`, inputs of type `tuple` are accepted
      and converted to `list`s. If `False`, an error is raised on tuple inputs.
    name: A `string`. The name of the argument being validated. This is only
      used to format error messages.

  Returns:
    A valid `list`.

  Raises:
    TypeError: When `value` does not meet the type requirements.
    ValueError: When `value` does not meet the length requirements.
  """
  # Handle tuples.
  if allow_tuples and isinstance(value, tuple):
    value = list(value)

  # Handle scalars.
  if broadcast_scalars:
    if ((element_type is not None and isinstance(value, element_type)) or
        (element_type is None and not isinstance(value, list))):
      value = [value] * (length if length is not None else 1)

  # We've handled tuples and scalars. If not a list by now, this is an error.
  if not isinstance(value, list):
    raise TypeError(
      f"Argument `{name}` must be a `list`, but received type: {type(value)}")

  # It's a list! Now check the length.
  if length is not None and not len(value) == length:
    raise ValueError(
      f"Argument `{name}` must be a `list` of length {length}, but received a "
      f"`list` of length {len(value)}")

  # It's a list with the correct length! Check element types.
  if element_type is not None:
    if not isinstance(element_type, (list, tuple)):
      element_types = (element_type,)
    else:
      element_types = element_type
    for element in value:
      if type(element) not in element_types:
        raise TypeError(
          f"Argument `{name}` must be a `list` of elements of type "
          f"`{element_type}`, but received type: `{type(element)}`")

  return value

def validate_enum(value, valid_values, name=None):
  """Validates that value is in a list of valid values.

  Args:
    value: The value to validate.
    valid_values: The list of valid values.
    name: The name of the argument being validated. This is only used to format
      error messages.

  Returns:
    A valid enum value.

  Raises:
    ValueError: If `value` is not in the list of valid values.
  """
  if value not in valid_values:
    raise ValueError(
      f"Argument `{name}` must be one of {valid_values}, but received value: "
      f"{value}")
  return value

def normalize_no_nan(tensor, ord='euclidean', axis=None, name=None):  # pylint: disable=redefined-builtin
  """Normalizes `tensor` along dimension `axis` using specified norm.

  Args:
    tensor: A `Tensor` of type `float32`, `float64`, `complex64`, `complex128`.
    ord: Order of the norm. Supported values are `'fro'`, `'euclidean'`, `1`,
      `2`, `np.inf` and any positive real number yielding the corresponding
      p-norm. Default is `'euclidean'` which is equivalent to Frobenius norm if
      `tensor` is a matrix and equivalent to 2-norm for vectors.
      Some restrictions apply: a) The Frobenius norm `'fro'` is not defined for
        vectors, b) If axis is a 2-tuple (matrix norm), only `'euclidean'`,
        '`fro'`, `1`, `2`, `np.inf` are supported. See the description of `axis`
        on how to compute norms for a batch of vectors or matrices stored in a
        tensor.
    axis: If `axis` is `None` (the default), the input is considered a vector
      and a single vector norm is computed over the entire set of values in the
      tensor, i.e. `norm(tensor, ord=ord)` is equivalent to
      `norm(reshape(tensor, [-1]), ord=ord)`. If `axis` is a Python integer, the
      input is considered a batch of vectors, and `axis` determines the axis in
      `tensor` over which to compute vector norms. If `axis` is a 2-tuple of
      Python integers it is considered a batch of matrices and `axis` determines
      the axes in `tensor` over which to compute a matrix norm.
      Negative indices are supported. Example: If you are passing a tensor that
        can be either a matrix or a batch of matrices at runtime, pass
        `axis=[-2,-1]` instead of `axis=None` to make sure that matrix norms are
        computed.
    name: The name of the op.

  Returns:
    A normalized `Tensor` with the same shape as `tensor`.
  """
  with tf.name_scope(name or 'normalize_no_nan'):
    norm = tf.norm(tensor, ord=ord, axis=axis, keepdims=True)
    norm = tf.cast(norm, tensor.dtype)
    return tf.math.divide_no_nan(tensor, norm)
  

def resize_with_crop_or_pad(tensor, shape, padding_mode='constant'):
  """Crops and/or pads a tensor to a target shape.

  Pads symmetrically or crops centrally the input tensor as necessary to achieve
  the requested shape.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.
    padding_mode: A `str`. Must be one of `'constant'`, `'reflect'` or
      `'symmetric'`.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape = tensor.shape
  input_shape_tensor = tf.shape(tensor)
  target_shape = shape
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(target_shape, tf.Tensor):
    target_shape = [-1] * (input_shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(shape)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor),
                                        tf.size(target_shape_tensor)),
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Pad the tensor.
  pad_left = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) // 2,
      0)
  pad_right = tf.where(
      target_shape_tensor >= 0,
      (tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) + 1) // 2,
      0)

  tensor = tf.pad(tensor, tf.transpose(tf.stack([pad_left, pad_right])), # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                  mode=padding_mode)

  # Crop the tensor.
  tensor = central_crop(tensor, target_shape)

  static_shape = _compute_static_output_shape(input_shape, target_shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor


def central_crop(tensor, shape):
  """Crop the central region of a tensor.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the region to crop. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The centrally cropped tensor.

  Raises:
    ValueError: If `shape` has a rank other than 1.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape_tensor = tf.shape(tensor)
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Static checks.
  if target_shape_tensor.shape.rank != 1:
    raise ValueError(f"`shape` must have rank 1. Received: {shape}")

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(shape, tf.Tensor):
    shape = [-1] * (tensor.shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(target_shape_tensor)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor), tf.size(shape)),
      tf.debugging.assert_less_equal(
          target_shape_tensor, tf.shape(tensor), message=(
              "Target shape cannot be greater than input shape."))
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Crop the tensor.
  slice_begin = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(input_shape_tensor - target_shape_tensor, 0) // 2,
      0)
  slice_size = tf.where(
      target_shape_tensor >= 0,
      tf.math.minimum(input_shape_tensor, target_shape_tensor),
      -1)
  tensor = tf.slice(tensor, slice_begin, slice_size)

  # Set static shape, if possible.
  static_shape = _compute_static_output_shape(tensor.shape, shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor

def _right_pad_or_crop(tensor, shape):
  """Pad or crop a tensor to the specified shape.

  The tensor will be padded and/or cropped in its right side.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. If the size of `shape` is
      smaller than the rank of `tensor`, it is assumed to refer to the innermost
      dimensions of `tensor`.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  # Get input and output shapes.
  input_shape = tf.shape(tensor)
  shape = tf.convert_to_tensor(shape, dtype=tf.dtypes.int32)

  # Normalize `shape`, which may have less dimensions than input. In this
  # case, `shape` is assumed to refer to the last dimensions in `x`.
  with tf.control_dependencies([tf.debugging.assert_less_equal(
      tf.size(shape), tf.size(input_shape))]):
    shape = tf.identity(shape)
  shape = tf.concat([input_shape[:tf.size(input_shape) - tf.size(shape)],
                     shape], 0)

  # Pad tensor with zeros.
  pad_sizes = tf.math.maximum(shape - input_shape, 0)
  pad_sizes = tf.expand_dims(pad_sizes, -1)
  pad_sizes = tf.concat([tf.zeros(pad_sizes.shape, dtype=tf.dtypes.int32),
                         pad_sizes], -1)
  tensor = tf.pad(tensor, pad_sizes, constant_values=0)

  # Crop tensor.
  begin = tf.zeros(shape.shape, dtype=tf.dtypes.int32)
  tensor = tf.slice(tensor, begin, shape)

  return tensor

def _compute_static_output_shape(input_shape, target_shape):
  """Compute the static output shape of a resize operation.

  Args:
    input_shape: The static shape of the input tensor.
    target_shape: The target shape.

  Returns:
    The static output shape.
  """
  output_shape = None

  if isinstance(target_shape, tf.Tensor):
    # If target shape is a tensor, we can't infer the output shape.
    return None

  # Get static tensor shape, after replacing -1 values by `None`.
  output_shape = tf.TensorShape(
      [s if s >= 0 else None for s in target_shape])

  # Complete any unspecified target dimensions with those of the
  # input tensor, if known.
  output_shape = tf.TensorShape(
      [s_target or s_input for (s_target, s_input) in zip(
          output_shape.as_list(), input_shape.as_list())])

  return output_shape


def broadcast_static_shapes(*shapes):
  """Computes the shape of a broadcast given known shapes.

  Like `tf.broadcast_static_shape`, but accepts any number of shapes.

  Args:
    *shapes: Two or more `TensorShapes`.

  Returns:
    A `TensorShape` representing the broadcasted shape.
  """
  bcast_shape = shapes[0]
  for shape in shapes[1:]:
    bcast_shape = tf.broadcast_static_shape(bcast_shape, shape)
  return bcast_shape


def broadcast_dynamic_shapes(*shapes):
  """Computes the shape of a broadcast given symbolic shapes.

  Like `tf.broadcast_dynamic_shape`, but accepts any number of shapes.

  Args:
    shapes: Two or more rank-1 integer `Tensors` representing the input shapes.

  Returns:
    A rank-1 integer `Tensor` representing the broadcasted shape.
  """
  bcast_shape = shapes[0]
  for shape in shapes[1:]:
    bcast_shape = tf.broadcast_dynamic_shape(bcast_shape, shape)
  return bcast_shape


def batch_shape(operator):
  """Returns the batch shape of an operator.

  Returns the static batch shape of the operator if fully known, otherwise
  returns the dynamic batch shape.

  Args:
    operator: A `tf.linalg.LinearOperator` or a `tfmri.convex.ConvexFunction`.

  Returns:
    A `tf.TensorShape` or a 1D integer `tf.Tensor` representing the batch shape
    of the operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator` or a
      `tfmri.convex.ConvexFunction`.
  """
  if not isinstance(operator, (tf.linalg.LinearOperator,
                               convex_ops.ConvexFunction)):
    raise ValueError(
        f"Input must be a `tf.linalg.LinearOperator` or a "
        f"`tfmri.convex.ConvexFunction`, but got: {type(operator)}")

  if operator.batch_shape.is_fully_defined():
    return operator.batch_shape

  return operator.batch_shape_tensor()


def domain_dimension(operator):
  """Retrieves the domain dimension of an operator.

  Args:
    operator: A `tf.linalg.LinearOperator` or a `tfmri.convex.ConvexFunction`.

  Returns:
    An int or scalar integer `tf.Tensor` representing the range dimension of the
    operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator` or a
    `tfmri.convex.ConvexFunction`.
  """
  if not isinstance(operator, (tf.linalg.LinearOperator,
                               convex_ops.ConvexFunction)):
    raise ValueError(f"Input must be a `tf.linalg.LinearOperator` or a "
                     f"`tfmri.convex.ConvexFunction`, "
                     f"but got: {type(operator)}")

  dimension = operator.domain_dimension
  if isinstance(dimension, tf.compat.v1.Dimension):
    dimension = dimension.value
  if dimension is not None:
    return dimension

  return operator.domain_dimension_tensor()



def range_dimension(operator):
  """Retrieves the range dimension of an operator.

  Args:
    operator: A `tf.linalg.LinearOperator`.

  Returns:
    An int or scalar integer `tf.Tensor` representing the range dimension of the
    operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator`.
  """
  if not isinstance(operator, tf.linalg.LinearOperator):
    raise ValueError(f"Input must be a `tf.linalg.LinearOperator`, "
                     f"but got: {type(operator)}")

  dimension = operator.range_dimension
  if isinstance(dimension, tf.compat.v1.Dimension):
    dimension = dimension.value
  if dimension is not None:
    return dimension

  return operator.range_dimension_tensor()

def object_shape(tensor):
  """Returns the shape of a tensor or an object.

  Args:
    tensor: A `tf.Tensor` or an object with a `shape_tensor` method.

  Returns:
    The shape of the input object.
  """
  if hasattr(tensor, 'shape_tensor'):
    return tensor.shape_tensor()
  return tf.shape(tensor)

def shrinkage(x, threshold, name=None):
  r"""Shrinkage operator.

  In the context of proximal gradient methods, this function is the proximal
  operator of :math:`f = \frac{1}{2}{\left\| x \right\|}_{2}^{2}`.

  Args:
    x: A `Tensor` of shape `[..., n]`.
    threshold: A `float`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` of shape `[..., n]` and same dtype as `x`.
  """
  with tf.name_scope(name or 'shrinkage'):
    x = tf.convert_to_tensor(x, name='x')
    threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
    one = tf.constant(1.0, dtype=x.dtype, name='one')
    return x / (one + threshold)
  

def view_as_real(x, stacked=True):
  """Returns a view of the input as a real tensor.

  For a complex-valued input tensor of shape `[M1, M2, ..., Mn]`:

  * If `stacked` is `True`, returns a new real-valued tensor of shape
    `[M1, M2, ..., Mn, 2]`, where the last axis has the real and imaginary
    components of the complex numbers.
  * If `stacked` is `False`, returns a new real-valued tensor of shape
    `[M1, M2, ..., 2 * Mn], where real and imaginary components are interleaved
    in the channel dimension.

  Args:
    x: A complex-valued `Tensor`.
    stacked: A `boolean`. If `True`, real and imaginary components are stacked
      along a new axis. If `False`, they are inserted into the channel axis.

  Returns:
    A real-valued `Tensor`.
  """
  x = tf.convert_to_tensor(x)
  static_shape = x.shape
  dynamic_shape = tf.shape(x)

  x = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)

  if not stacked:
    dynamic_shape = tf.concat([dynamic_shape[:-1], 2 * dynamic_shape[-1:]], 0)
    if static_shape[-1] is None:
      static_shape = static_shape[:-1] + [None]
    else:
      static_shape = static_shape[:-1] + [2 * static_shape[-1]]
    x = tf.reshape(x, dynamic_shape)
    x = tf.ensure_shape(x, static_shape)

  return x


def view_as_complex(x, stacked=True):
  """Returns a view of the input as a complex tensor.

  Returns a new complex-valued input tensor of shape `[M1, M2, ..., Mn]`:

  * If `stacked` is `True`, expects a real-valued tensor of shape
    `[M1, M2, ..., Mn, 2]`, where the last axis has the real and imaginary
    components of the complex numbers.
  * If `stacked` is `False`, expects a real-valued tensor of shape
    `[M1, M2, ..., 2 * Mn], where real and imaginary components are interleaved
    in the channel dimension.

  Args:
    x: A real-valued `Tensor`.
    stacked: A `boolean`. If `True`, real and imaginary components are expected
      to be stacked in their own axis. If `False`, they are expected to be
      interleaved in the channel dimension.

  Returns:
    A complex-valued `Tensor`.
  """
  x = tf.convert_to_tensor(x)
  if not stacked:
    x_shape = tf.shape(x)
    x_shape = tf.concat([x_shape[:-1], [x_shape[-1] // 2], [2]], 0)
    x = tf.reshape(x, x_shape)
  checks = [tf.debugging.assert_equal(tf.shape(x)[-1], 2, message=(
      f"Could not interpret input tensor as complex. Last dimension must be 2, "
      f"but got {tf.shape(x)[-1]}. Perhaps you need to set `stacked` to "
      f"`False`?"))]
  with tf.control_dependencies(checks):
    x = tf.identity(x)
  x = tf.complex(x[..., 0], x[..., 1])
  return x

def make_val_and_grad_fn(value_fn):
  """Function decorator to compute both function value and gradient.

  Turns function `value_fn` that evaluates and returns a `Tensor` with the value
  of the function evaluated at the input point into one that returns a tuple of
  two `Tensors` with the value and the gradient of the defined function
  evaluated at the input point.

  This is useful for constructing functions for optimization.

  Args:
    value_fn: A Python function to decorate.

  Returns:
    The decorated function.
  """
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad

def hann(arg, name=None):
  """Calculate a Hann window at the specified coordinates.

  The domain of the window is `[-pi, pi]`. Outside this range, its value is 0.
  The center of the window is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hann window at the specified coordinates.
  """
  with tf.name_scope(name or 'hann'):
    return _raised_cosine(arg, 0.5, 0.5)


def hamming(arg, name=None):
  """Calculate a Hamming window at the specified coordinates.

  The domain of the window is `[-pi, pi]`. Outside this range, its value is 0.
  The center of the window is at 0.

  Args:
    arg: Input tensor.
    name: Name to use for the scope.

  Returns:
    The value of a Hamming window at the specified coordinates.
  """
  with tf.name_scope(name or 'hamming'):
    return _raised_cosine(arg, 0.54, 0.46)


def _raised_cosine(arg, a, b):
  """Helper function for computing a raised cosine window.

  Args:
    arg: Input tensor.
    a: The alpha parameter to the raised cosine filter.
    b: The beta parameter to the raised cosine filter.

  Returns:
    A `Tensor` of shape `arg.shape`.
  """
  arg = tf.convert_to_tensor(arg)
  return tf.where(tf.math.abs(arg) <= np.pi,
                  a - b * tf.math.cos(arg + np.pi), 0.0)



def atanfilt(arg, cutoff=np.pi, beta=100.0, name=None):
  """Calculate an inverse tangent filter window at the specified coordinates.

  This window has infinite domain.

  Args:
    arg: Input tensor.
    cutoff: A `float` in the range [0, pi]. The cutoff frequency of the filter.
    beta: A `float`. The beta parameter of the filter.
    name: Name to use for the scope.

  Returns:
    A `Tensor` of shape `arg.shape`.

  References:
    .. [1] Pruessmann, K.P., Weiger, M., Börnert, P. and Boesiger, P. (2001),
      Advances in sensitivity encoding with arbitrary k-space trajectories.
      Magn. Reson. Med., 46: 638-651. https://doi.org/10.1002/mrm.1241
  """
  with tf.name_scope(name or 'atanfilt'):
    arg = tf.math.abs(tf.convert_to_tensor(arg))
    return 0.5 + (1.0 / np.pi) * tf.math.atan(beta * (cutoff - arg) / cutoff)


def verify_compatible_trajectory(kspace, traj):
  """Verifies that a trajectory is compatible with the given k-space.

  Args:
    kspace: A `Tensor`.
    traj: A `Tensor`.

  Returns:
    A tuple containing valid `kspace` and `traj` tensors.

  Raises:
    TypeError: If `kspace` and `traj` have incompatible dtypes.
    ValueError: If `kspace` and `traj` do not have the same number of samples
      or have incompatible batch shapes.
  """
  kspace = tf.convert_to_tensor(kspace, name='kspace')
  traj = tf.convert_to_tensor(traj, name='traj')

  # Check dtype.
  if traj.dtype != kspace.dtype.real_dtype:
    raise TypeError(
        f"kspace and trajectory have incompatible dtypes: "
        f"{kspace.dtype} and {traj.dtype}")

  # Check number of samples (static).
  if not kspace.shape[-1:].is_compatible_with(traj.shape[-2:-1]):
    raise ValueError(
        f"kspace and trajectory must have the same number of samples, but got "
        f"{kspace.shape[-1]} and {traj.shape[-2]}, respectively")
  # Check number of samples (dynamic).
  kspace_shape, traj_shape = tf.shape(kspace), tf.shape(traj)
  checks = [
      tf.debugging.assert_equal(
          kspace_shape[-1], traj_shape[-2],
          message="kspace and trajectory must have the same number of samples")
  ]
  with tf.control_dependencies(checks):
    kspace, traj = tf.identity_n([kspace, traj])

  # Check batch shapes (static).
  try:
    tf.broadcast_static_shape(kspace.shape[:-1], traj.shape[:-2])
  except ValueError as err:
    raise ValueError(
        f"kspace and trajectory have incompatible batch shapes, "
        f"got {kspace.shape[:-1]} and {traj.shape[:-2]}, respectively") from err
  # TODO(jmontalt): Check batch shapes (dynamic).

  return kspace, traj

def meshgrid(*args):
  """Return coordinate matrices from coordinate vectors.

  Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
  fields over N-D grids, given one-dimensional coordinate arrays
  `x1, x2, ..., xn`.

  .. note::
    Similar to `tf.meshgrid`, but uses matrix indexing and returns a stacked
    tensor (along axis -1) instead of a list of tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M1, M2, ..., Mn, N]`, where `N` is the number of
    tensors in `args` and `Mi = tf.size(args[i])`.
  """
  return tf.stack(tf.meshgrid(*args, indexing='ij'), axis=-1)


def filter_kspace(kspace,
                  trajectory=None,
                  filter_fn='hamming',
                  filter_rank=None,
                  filter_kwargs=None):
  """Filter *k*-space.

  Multiplies *k*-space by a filtering function.

  Args:
    kspace: A `Tensor` of any shape. The input *k*-space.
    trajectory: A `Tensor` of shape `kspace.shape + [N]`, where `N` is the
      number of spatial dimensions. If `None`, `kspace` is assumed to be
      Cartesian.
    filter_fn: A `str` (one of `'hamming'`, `'hann'` or `'atanfilt'`) or a
      callable that accepts a coordinate array and returns corresponding filter
      values.
    filter_rank: An `int`. The rank of the filter. Only relevant if *k*-space is
      Cartesian. Defaults to `kspace.shape.rank`.
    filter_kwargs: A `dict`. Additional keyword arguments to pass to the
      filtering function.

  Returns:
    A `Tensor` of shape `kspace.shape`. The filtered *k*-space.
  """
  kspace = tf.convert_to_tensor(kspace)
  if trajectory is not None:
    kspace, trajectory = verify_compatible_trajectory(
        kspace, trajectory)

  # Make a "trajectory" for Cartesian k-spaces.
  is_cartesian = trajectory is None
  if is_cartesian:
    filter_rank = filter_rank or kspace.shape.rank
    vecs = [tf.linspace(-np.pi, np.pi - (2.0 * np.pi / s), s)
            for s in kspace.shape[-filter_rank:]]  # pylint: disable=invalid-unary-operand-type
    trajectory = meshgrid(*vecs)

  if not callable(filter_fn):
    # filter_fn not a callable, so should be an enum value. Get the
    # corresponding function.
    filter_fn = utils.validate_enum(
        filter_fn, valid_values={'hamming', 'hann', 'atanfilt'},
        name='filter_fn')
    filter_fn = {
        'hamming': hamming,
        'hann': hann,
        'atanfilt': atanfilt
    }[filter_fn]
  filter_kwargs = filter_kwargs or {}

  traj_norm = tf.norm(trajectory, axis=-1)
  return kspace * tf.cast(filter_fn(traj_norm, **filter_kwargs), kspace.dtype)



