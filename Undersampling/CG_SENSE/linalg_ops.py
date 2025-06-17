import collections
import functools
import CG_SENSE.utils as utils
import CG_SENSE.fft as fft
import CG_SENSE.convex_ops  as convex_ops
import tensorflow_probability as tfp

import abc
import contextlib

import tensorflow as tf

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
from tfkbnufft.mri.dcomp_calc import calculate_radial_dcomp_tf, calculate_density_compensator

class LinearOperatorAddition(tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  """Adds one or more `LinearOperators`.

  This operator adds one or more linear operators `op1 + op2 + ... + opJ`,
  building a new `LinearOperator` with action defined by:

  ```
  op_addition(x) := op1(x) + op2(x) + op3(x)
  ```

  If `opj` acts like [batch] matrix `Aj`, then `op_addition` acts like the
  [batch] matrix formed with the addition `A1 + A2 + ... + AJ`.

  If each `opj` has shape `batch_shape_j + [M, N]`, then the addition operator
  has shape equal to `broadcast_batch_shape + [M, N]`, where
  `broadcast_batch_shape` is the mutual broadcast of `batch_shape_j`,
  `j = 1, ..., J`, assuming the intermediate batch shapes broadcast.

  ```python
  # Create a 2 x 2 linear operator composed of two 2 x 2 operators.
  operator_1 = LinearOperatorFullMatrix([[1., 2.], [3., 4.]])
  operator_2 = LinearOperatorFullMatrix([[1., 0.], [0., 1.]])
  operator = LinearOperatorAddition([operator_1, operator_2])

  operator.to_dense()
  ==> [[2., 2.]
       [3., 5.]]

  operator.shape
  ==> [2, 2]
  ```

  #### Performance

  The performance of `LinearOperatorAddition` on any operation is equal to
  the sum of the individual operators' operations.


  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.

  Args:
    operators: Iterable of `LinearOperator` objects, each with
      the same shape and dtype.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_p_`.

  Raises:
    TypeError: If all operators do not have the same `dtype`.
    ValueError: If `operators` is empty.
  """
  def __init__(self,
               operators,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    """Initialize a `LinearOperatorAddition`."""
    parameters = dict(
        operators=operators,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)

    # Validate operators.
    tf.debugging.assert_proper_iterable(operators)
    operators = list(operators)
    if not operators:
      raise ValueError(
          "Expected a non-empty list of operators. Found: %s" % operators)
    self._operators = operators

    # Validate dtype.
    dtype = operators[0].dtype
    for operator in operators:
      if operator.dtype != dtype:
        name_type = (str((o.name, o.dtype)) for o in operators)
        raise TypeError(
            "Expected all operators to have the same dtype.  Found %s"
            % "   ".join(name_type))

    # Infer operator properties.
    if is_self_adjoint is None:
      # If all operators are self-adjoint, so is the sum.
      if all(operator.is_self_adjoint for operator in operators):
        is_self_adjoint = True
    if is_positive_definite is None:
      # If all operators are positive definite, so is the sum.
      if all(operator.is_positive_definite for operator in operators):
        is_positive_definite = True
    if is_non_singular is None:
      # A positive definite operator is always non-singular.
      if is_positive_definite:
        is_non_singular = True
    if is_square is None:
      # If all operators are square, so is the sum.
      if all(operator.is_square for operator in operators):
        is_square=True

    if name is None:
      name = "_p_".join(operator.name for operator in operators)
    with tf.name_scope(name):
      super().__init__(
          dtype=dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def operators(self):
    return self._operators

  def _shape(self):
    # Get final matrix shape.
    domain_dimension = self.operators[0].domain_dimension
    range_dimension = self.operators[1].range_dimension
    for operator in self.operators[1:]:
      domain_dimension.assert_is_compatible_with(operator.domain_dimension)
      range_dimension.assert_is_compatible_with(operator.range_dimension)

    matrix_shape = tf.TensorShape([range_dimension, domain_dimension])

    # Get broadcast batch shape.
    # tf.broadcast_static_shape checks for compatibility.
    batch_shape = self.operators[0].batch_shape
    for operator in self.operators[1:]:
      batch_shape = tf.broadcast_static_shape(
          batch_shape, operator.batch_shape)

    return batch_shape.concatenate(matrix_shape)

  def _shape_tensor(self):
    # Avoid messy broadcasting if possible.
    if self.shape.is_fully_defined():
      return tf.convert_to_tensor(
          self.shape.as_list(), dtype=tf.dtypes.int32, name="shape")

    # Don't check the matrix dimensions.  That would add unnecessary Asserts to
    # the graph.  Things will fail at runtime naturally if shapes are
    # incompatible.
    matrix_shape = tf.stack([
        self.operators[0].range_dimension_tensor(),
        self.operators[0].domain_dimension_tensor()
    ])

    # Dummy Tensor of zeros.  Will never be materialized.
    zeros = tf.zeros(shape=self.operators[0].batch_shape_tensor())
    for operator in self.operators[1:]:
      zeros += tf.zeros(shape=operator.batch_shape_tensor())
    batch_shape = tf.shape(zeros)

    return tf.concat((batch_shape, matrix_shape), 0)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    result = self.operators[0].matmul(
        x, adjoint=adjoint, adjoint_arg=adjoint_arg)
    for operator in self.operators[1:]:
      result += operator.matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)
    return result

  @property
  def _composite_tensor_fields(self):
    return ("operators",)


class LinalgImagingMixin(tf.linalg.LinearOperator):
  """Mixin for linear operators meant to operate on images."""
  def transform(self, x, adjoint=False, name="transform"):
    """Transform a batch of images.

    Applies this operator to a batch of non-vectorized images `x`.

    Args:
      x: A `Tensor` with compatible shape and same dtype as `self`.
      adjoint: A `boolean`. If `True`, transforms the input using the adjoint
        of the operator, instead of the operator itself.
      name: A name for this operation.

    Returns:
      The transformed `Tensor` with the same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tf.convert_to_tensor(x, name="x")
      self._check_input_dtype(x)
      input_shape = self.range_shape if adjoint else self.domain_shape
      input_shape.assert_is_compatible_with(x.shape[-input_shape.rank:])  # pylint: disable=invalid-unary-operand-type
      return self._transform(x, adjoint=adjoint)

  @property
  def domain_shape(self):
    """Domain shape of this linear operator."""
    return self._domain_shape()

  @property
  def range_shape(self):
    """Range shape of this linear operator."""
    return self._range_shape()

  def domain_shape_tensor(self, name="domain_shape_tensor"):
    """Domain shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.domain_shape.is_fully_defined():
        return utils.convert_shape_to_tensor(self.domain_shape.as_list())
      return self._domain_shape_tensor()

  def range_shape_tensor(self, name="range_shape_tensor"):
    """Range shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.range_shape.is_fully_defined():
        return utils.convert_shape_to_tensor(self.range_shape.as_list())
      return self._range_shape_tensor()

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    """Batch shape of this linear operator, determined at runtime."""
    with self._name_scope(name):  # pylint: disable=not-callable
      if self.batch_shape.is_fully_defined():
        return utils.convert_shape_to_tensor(self.batch_shape.as_list())
      return self._batch_shape_tensor()

  def adjoint(self, name="adjoint"):
    """Returns the adjoint of this linear operator.

    The returned operator is a valid `LinalgImagingMixin` instance.

    Calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name: A name for this operation.

    Returns:
      A `LinearOperator` derived from `LinalgImagingMixin`, which
      represents the adjoint of this linear operator.
    """
    if self.is_self_adjoint:
      return self
    with self._name_scope(name):  # pylint: disable=not-callable
      return LinearOperatorAdjoint(self)

  H = property(adjoint, None)

  @abc.abstractmethod
  def _transform(self, x, adjoint=False):
    # Subclasses must override this method.
    raise NotImplementedError("Method `_transform` is not implemented.")

  def _matvec(self, x, adjoint=False):
    # Default implementation of `_matvec` for imaging operator. The vectorized
    # input `x` is first expanded to the its full shape, then transformed, then
    # vectorized again. Typically subclasses should not need to override this
    # method.
    x = self.expand_range_dimension(x) if adjoint else \
        self.expand_domain_dimension(x)
    x = self._transform(x, adjoint=adjoint)
    x = self.flatten_domain_shape(x) if adjoint else \
        self.flatten_range_shape(x)
    return x

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # Default implementation of `matmul` for imaging operator. If outer
    # dimension of argument is 1, call `matvec`. Otherwise raise an error.
    # Typically subclasses should not need to override this method.
    arg_outer_dim = -2 if adjoint_arg else -1

    if x.shape[arg_outer_dim] != 1:
      raise ValueError(
        f"`{self.__class__.__name__}` does not support matrix multiplication.")

    x = tf.squeeze(x, axis=arg_outer_dim)
    x = self.matvec(x, adjoint=adjoint)
    x = tf.expand_dims(x, axis=arg_outer_dim)
    return x

  @abc.abstractmethod
  def _domain_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  @abc.abstractmethod
  def _range_shape(self):
    # Users must override this method.
    return tf.TensorShape(None)

  def _batch_shape(self):
    # Users should override this method if this operator has a batch shape.
    return tf.TensorShape([])

  def _domain_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic domain
    # shape.
    raise NotImplementedError("_domain_shape_tensor is not implemented.")

  def _range_shape_tensor(self):
    # Users should override this method if they need to provide a dynamic range
    # shape.
    raise NotImplementedError("_range_shape_tensor is not implemented.")

  def _batch_shape_tensor(self):  # pylint: disable=arguments-differ
    # Users should override this method if they need to provide a dynamic batch
    # shape.
    return tf.constant([], dtype=tf.dtypes.int32)

  def _shape(self):
    # Default implementation of `_shape` for imaging operators. Typically
    # subclasses should not need to override this method.
    return self._batch_shape() + tf.TensorShape(
        [self.range_shape.num_elements(),
         self.domain_shape.num_elements()])

  def _shape_tensor(self):
    # Default implementation of `_shape_tensor` for imaging operators. Typically
    # subclasses should not need to override this method.
    return tf.concat([self.batch_shape_tensor(),
                      [tf.size(self.range_shape_tensor()),
                       tf.size(self.domain_shape_tensor())]], 0)

  def flatten_domain_shape(self, x):
    """Flattens `x` to match the domain dimension of this operator.

    Args:
      x: A `Tensor`. Must have shape `[...] + self.domain_shape`.

    Returns:
      The flattened `Tensor`. Has shape `[..., self.domain_dimension]`.
    """
    # pylint: disable=invalid-unary-operand-type
    self.domain_shape.assert_is_compatible_with(
        x.shape[-self.domain_shape.rank:])

    batch_shape = x.shape[:-self.domain_shape.rank]
    batch_shape_tensor = tf.shape(x)[:-self.domain_shape.rank]

    output_shape = batch_shape + self.domain_dimension
    output_shape_tensor = tf.concat(
        [batch_shape_tensor, [self.domain_dimension_tensor()]], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def flatten_range_shape(self, x):
    """Flattens `x` to match the range dimension of this operator.

    Args:
      x: A `Tensor`. Must have shape `[...] + self.range_shape`.

    Returns:
      The flattened `Tensor`. Has shape `[..., self.range_dimension]`.
    """
    # pylint: disable=invalid-unary-operand-type
    self.range_shape.assert_is_compatible_with(
        x.shape[-self.range_shape.rank:])

    batch_shape = x.shape[:-self.range_shape.rank]
    batch_shape_tensor = tf.shape(x)[:-self.range_shape.rank]

    output_shape = batch_shape + self.range_dimension
    output_shape_tensor = tf.concat(
        [batch_shape_tensor, [self.range_dimension_tensor()]], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def expand_domain_dimension(self, x):
    """Expands `x` to match the domain shape of this operator.

    Args:
      x: A `Tensor`. Must have shape `[..., self.domain_dimension]`.

    Returns:
      The expanded `Tensor`. Has shape `[...] + self.domain_shape`.
    """
    self.domain_dimension.assert_is_compatible_with(x.shape[-1])

    batch_shape = x.shape[:-1]
    batch_shape_tensor = tf.shape(x)[:-1]

    output_shape = batch_shape + self.domain_shape
    output_shape_tensor = tf.concat([
        batch_shape_tensor, self.domain_shape_tensor()], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

  def expand_range_dimension(self, x):
    """Expands `x` to match the range shape of this operator.

    Args:
      x: A `Tensor`. Must have shape `[..., self.range_dimension]`.

    Returns:
      The expanded `Tensor`. Has shape `[...] + self.range_shape`.
    """
    self.range_dimension.assert_is_compatible_with(x.shape[-1])

    batch_shape = x.shape[:-1]
    batch_shape_tensor = tf.shape(x)[:-1]

    output_shape = batch_shape + self.range_shape
    output_shape_tensor = tf.concat([
        batch_shape_tensor, self.range_shape_tensor()], 0)

    x = tf.reshape(x, output_shape_tensor)
    return tf.ensure_shape(x, output_shape)

class LinearOperatorAdjoint(LinalgImagingMixin,  # pylint: disable=abstract-method
                            tf.linalg.LinearOperatorAdjoint):
  """Linear operator representing the adjoint of another operator.

  `LinearOperatorAdjoint` is initialized with an operator :math:`A` and
  represents its adjoint :math:`A^H`.

  .. note:
    Similar to `tf.linalg.LinearOperatorAdjoint`_, but with imaging extensions.

  Args:
    operator: A `LinearOperator`.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`. Default is `operator.name +
      "_adjoint"`.

  .. _tf.linalg.LinearOperatorAdjoint: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorAdjoint
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    return self.operator._transform(x, adjoint=(not adjoint))

  def _domain_shape(self):
    return self.operator.range_shape

  def _range_shape(self):
    return self.operator.domain_shape

  def _batch_shape(self):
    return self.operator.batch_shape

  def _domain_shape_tensor(self):
    return self.operator.range_shape_tensor()

  def _range_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()
  

class LinearOperator(LinalgImagingMixin, tf.linalg.LinearOperator):  # pylint: disable=abstract-method
  r"""Base class defining a [batch of] linear operator[s].

  Provides access to common matrix operations without the need to materialize
  the matrix.

  This operator is similar to `tf.linalg.LinearOperator`_, but has additional
  methods to simplify operations on images, while maintaining compatibility
  with the TensorFlow linear algebra framework.

  Inputs and outputs to this linear operator or its subclasses may have
  meaningful non-vectorized N-D shapes. Thus this class defines the additional
  properties `domain_shape` and `range_shape` and the methods
  `domain_shape_tensor` and `range_shape_tensor`. These enrich the information
  provided by the built-in properties `shape`, `domain_dimension`,
  `range_dimension` and methods `domain_dimension_tensor` and
  `range_dimension_tensor`, which only have information about the vectorized 1D
  shapes.

  Subclasses of this operator must define the methods `_domain_shape` and
  `_range_shape`, which return the static domain and range shapes of the
  operator. Optionally, subclasses may also define the methods
  `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
  domain and range shapes of the operator. These two methods will only be called
  if `_domain_shape` and `_range_shape` do not return fully defined static
  shapes.

  Subclasses must define the abstract method `_transform`, which
  applies the operator (or its adjoint) to a [batch of] images. This internal
  method is called by `transform`. In general, subclasses of this operator
  should not define the methods `_matvec` or `_matmul`. These have default
  implementations which call `_transform`.

  Operators derived from this class may be used in any of the following ways:

  1. Using method `transform`, which expects a full-shaped input and returns
     a full-shaped output, i.e. a tensor with shape `[...] + shape`, where
     `shape` is either the `domain_shape` or the `range_shape`. This method is
     unique to operators derived from this class.
  2. Using method `matvec`, which expects a vectorized input and returns a
     vectorized output, i.e. a tensor with shape `[..., n]` where `n` is
     either the `domain_dimension` or the `range_dimension`. This method is
     part of the TensorFlow linear algebra framework.
  3. Using method `matmul`, which expects matrix inputs and returns matrix
     outputs. Note that a matrix is just a column vector in this context, i.e.
     a tensor with shape `[..., n, 1]`, where `n` is either the
     `domain_dimension` or the `range_dimension`. Matrices which are not column
     vectors (i.e. whose last dimension is not 1) are not supported. This
     method is part of the TensorFlow linear algebra framework.

  Operators derived from this class may also be used with the functions
  `tf.linalg.matvec`_ and `tf.linalg.matmul`_, which will call the
  corresponding methods.

  This class also provides the convenience functions `flatten_domain_shape` and
  `flatten_range_shape` to flatten full-shaped inputs/outputs to their
  vectorized form. Conversely, `expand_domain_dimension` and
  `expand_range_dimension` may be used to expand vectorized inputs/outputs to
  their full-shaped form.

  **Subclassing**

  Subclasses must always define `_transform`, which implements this operator's
  functionality (and its adjoint). In general, subclasses should not define the
  methods `_matvec` or `_matmul`. These have default implementations which call
  `_transform`.

  Subclasses must always define `_domain_shape`
  and `_range_shape`, which return the static domain/range shapes of the
  operator. If the subclassed operator needs to provide dynamic domain/range
  shapes and the static shapes are not always fully-defined, it must also define
  `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
  domain/range shapes of the operator. In general, subclasses should not define
  the methods `_shape` or `_shape_tensor`. These have default implementations.

  If the subclassed operator has a non-scalar batch shape, it must also define
  `_batch_shape` which returns the static batch shape. If the static batch shape
  is not always fully-defined, the subclass must also define
  `_batch_shape_tensor`, which returns the dynamic batch shape.

  Args:
    dtype: The `tf.dtypes.DType` of the matrix that this operator represents.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose. If `dtype` is real, this is equivalent to being symmetric.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.

  .. _tf.linalg.LinearOperator: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperator
  .. _tf.linalg.matvec: https://www.tensorflow.org/api_docs/python/tf/linalg/matvec
  .. _tf.linalg.matmul: https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
  """

class LinearOperatorMRI(LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing an MRI encoding matrix.

  The MRI operator, :math:`A`, maps a [batch of] images, :math:`x` to a
  [batch of] measurement data (*k*-space), :math:`b`.

  .. math::
    A x = b

  This object may represent an undersampled MRI operator and supports
  Cartesian and non-Cartesian *k*-space sampling. The user may provide a
  sampling `mask` to represent an undersampled Cartesian operator, or a
  `trajectory` to represent a non-Cartesian operator.

  This object may represent a multicoil MRI operator by providing coil
  `sensitivities`. Note that `mask`, `trajectory` and `density` should never
  have a coil dimension, including in the case of multicoil imaging. The coil
  dimension will be handled automatically.

  The domain shape of this operator is `extra_shape + image_shape`. The range
  of this operator is `extra_shape + [num_coils] + image_shape`, for
  Cartesian imaging, or `extra_shape + [num_coils] + [num_samples]`, for
  non-Cartesian imaging. `[num_coils]` is optional and only present for
  multicoil operators. This operator supports batches of images and will
  vectorize operations when possible.

  Args:
    image_shape: A `tf.TensorShape` or a list of `ints`. The shape of the images
      that this operator acts on. Must have length 2 or 3.
    extra_shape: An optional `tf.TensorShape` or list of `ints`. Additional
      dimensions that should be included within the operator domain. Note that
      `extra_shape` is not needed to reconstruct independent batches of images.
      However, it is useful when this operator is used as part of a
      reconstruction that performs computation along non-spatial dimensions,
      e.g. for temporal regularization. Defaults to `None`.
    mask: An optional `tf.Tensor` of type `tf.bool`. The sampling mask. Must
      have shape `[..., *S]`, where `S` is the `image_shape` and `...` is
      the batch shape, which can have any number of dimensions. If `mask` is
      passed, this operator represents an undersampled MRI operator.
    trajectory: An optional `tf.Tensor` of type `float32` or `float64`. Must
      have shape `[..., M, N]`, where `N` is the rank (number of spatial
      dimensions), `M` is the number of samples in the encoded space and `...`
      is the batch shape, which can have any number of dimensions. If
      `trajectory` is passed, this operator represents a non-Cartesian MRI
      operator.
    density: An optional `tf.Tensor` of type `float32` or `float64`. The
      sampling densities. Must have shape `[..., M]`, where `M` is the number of
      samples and `...` is the batch shape, which can have any number of
      dimensions. This input is only relevant for non-Cartesian MRI operators.
      If passed, the non-Cartesian operator will include sampling density
      compensation. If `None`, the operator will not perform sampling density
      compensation.
    sensitivities: An optional `tf.Tensor` of type `complex64` or `complex128`.
      The coil sensitivity maps. Must have shape `[..., C, *S]`, where `S`
      is the `image_shape`, `C` is the number of coils and `...` is the batch
      shape, which can have any number of dimensions.
    phase: An optional `tf.Tensor` of type `float32` or `float64`. A phase
      estimate for the image. If provided, this operator will be
      phase-constrained.
    fft_norm: FFT normalization mode. Must be `None` (no normalization)
      or `'ortho'`. Defaults to `'ortho'`.
    sens_norm: A `boolean`. Whether to normalize coil sensitivities. Defaults to
      `True`.
    dynamic_domain: A `str`. The domain of the dynamic dimension, if present.
      Must be one of `'time'` or `'frequency'`. May only be provided together
      with a non-scalar `extra_shape`. The dynamic dimension is the last
      dimension of `extra_shape`. The `'time'` mode (default) should be
      used for regular dynamic reconstruction. The `'frequency'` mode should be
      used for reconstruction in x-f space.
    dtype: A `tf.dtypes.DType`. The dtype of this operator. Must be `complex64`
      or `complex128`. Defaults to `complex64`.
    name: An optional `str`. The name of this operator.
  """
  def __init__(self,
               image_shape,
               extra_shape=None,
               mask=None,
               trajectory=None,
               density=None,
               sensitivities=None,
               phase=None,
               fft_norm='ortho',
               sens_norm=True,
               dynamic_domain=None,
               dtype=tf.complex64,
               name=None):
    # pylint: disable=invalid-unary-operand-type
    parameters = dict(
        image_shape=image_shape,
        extra_shape=extra_shape,
        mask=mask,
        trajectory=trajectory,
        density=density,
        sensitivities=sensitivities,
        phase=phase,
        fft_norm=fft_norm,
        sens_norm=sens_norm,
        dynamic_domain=dynamic_domain,
        dtype=dtype,
        name=name)

    # Set dtype.
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.complex64, tf.complex128):
      raise ValueError(
          f"`dtype` must be `complex64` or `complex128`, but got: {str(dtype)}")

    # Set image shape, rank and extra shape.
    image_shape = tf.TensorShape(image_shape)
    rank = image_shape.rank
    if rank not in (2, 3):
      raise ValueError(
          f"Rank must be 2 or 3, but got: {rank}")
    if not image_shape.is_fully_defined():
      raise ValueError(
          f"`image_shape` must be fully defined, but got {image_shape}")
    self._rank = rank
    self._image_shape = image_shape
    self._image_axes = list(range(-self._rank, 0))  # pylint: disable=invalid-unary-operand-type
    self._extra_shape = tf.TensorShape(extra_shape or [])

    # Set initial batch shape, then update according to inputs.
    batch_shape = self._extra_shape
    batch_shape_tensor = utils.convert_shape_to_tensor(batch_shape)

    # Set sampling mask after checking dtype and static shape.
    if mask is not None:
      mask = tf.convert_to_tensor(mask)
      if mask.dtype != tf.bool:
        raise TypeError(
            f"`mask` must have dtype `bool`, but got: {str(mask.dtype)}")
      if not mask.shape[-self._rank:].is_compatible_with(self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `mask` to be compatible with "
            f"{self._image_shape}], but got: {mask.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, mask.shape[:-self._rank])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(mask)[:-self._rank])
    self._mask = mask

    # Set sampling trajectory after checking dtype and static shape.
    if trajectory is not None:
      if mask is not None:
        raise ValueError("`mask` and `trajectory` cannot be both passed.")
      trajectory = tf.convert_to_tensor(trajectory)
      if trajectory.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `trajectory` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(trajectory.dtype)}")
      if trajectory.shape[-1] != self._rank:
        raise ValueError(
            f"Expected the last dimension of `trajectory` to be "
            f"{self._rank}, but got {trajectory.shape[-1]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, trajectory.shape[:-2])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(trajectory)[:-2])
    self._trajectory = trajectory

# Set sampling density after checking dtype and static shape.
    if density is not None:
      if self._trajectory is None:
        raise ValueError("`density` must be passed with `trajectory`.")
      density = tf.convert_to_tensor(density)
      if density.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `density` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(density.dtype)}")
      if density.shape[-1] != self._trajectory.shape[-2]:
        raise ValueError(
            f"Expected the last dimension of `density` to be "
            f"{self._trajectory.shape[-2]}, but got {density.shape[-1]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, density.shape[:-1])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
        batch_shape_tensor, tf.shape(density)[:-1])
    self._density = density

    # Set sensitivity maps after checking dtype and static shape.
    if sensitivities is not None:
      sensitivities = tf.convert_to_tensor(sensitivities)
      if sensitivities.dtype != dtype:
        raise TypeError(
            f"Expected `sensitivities` to have dtype `{str(dtype)}`, but got: "
            f"{str(sensitivities.dtype)}")
      if not sensitivities.shape[-self._rank:].is_compatible_with(
          self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `sensitivities` to be "
            f"compatible with {self._image_shape}, but got: "
            f"{sensitivities.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, sensitivities.shape[:-(self._rank + 1)])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(sensitivities)[:-(self._rank + 1)])
    self._sensitivities = sensitivities

    if phase is not None:
      phase = tf.convert_to_tensor(phase)
      if phase.dtype != dtype.real_dtype:
        raise TypeError(
            f"Expected `phase` to have dtype `{str(dtype.real_dtype)}`, "
            f"but got: {str(phase.dtype)}")
      if not phase.shape[-self._rank:].is_compatible_with(
          self._image_shape):
        raise ValueError(
            f"Expected the last dimensions of `phase` to be "
            f"compatible with {self._image_shape}, but got: "
            f"{phase.shape[-self._rank:]}")
      batch_shape = tf.broadcast_static_shape(
          batch_shape, phase.shape[:-self._rank])
      batch_shape_tensor = tf.broadcast_dynamic_shape(
          batch_shape_tensor, tf.shape(phase)[:-self._rank])
    self._phase = phase

    # Set batch shapes.
    self._batch_shape_value = batch_shape
    self._batch_shape_tensor_value = batch_shape_tensor

    # If multicoil, add coil dimension to mask, trajectory and density.
    if self._sensitivities is not None:
      if self._mask is not None:
        self._mask = tf.expand_dims(self._mask, axis=-(self._rank + 1))
      if self._trajectory is not None:
        self._trajectory = tf.expand_dims(self._trajectory, axis=-3)
      if self._density is not None:
        self._density = tf.expand_dims(self._density, axis=-2)
      if self._phase is not None:
        self._phase = tf.expand_dims(self._phase, axis=-(self._rank + 1))

    # Save some tensors for later use during computation.
    if self._mask is not None:
      self._mask_linop_dtype = tf.cast(self._mask, dtype)
    if self._density is not None:
      self._dens_weights_sqrt = tf.cast(
          tf.math.sqrt(tf.math.reciprocal_no_nan(self._density)), dtype)
    if self._phase is not None:
      self._phase_rotator = tf.math.exp(
          tf.complex(tf.constant(0.0, dtype=phase.dtype), phase))

    # Set normalization.
    self._fft_norm = utils.validate_enum(
        fft_norm, {None, 'ortho'}, 'fft_norm')
    if self._fft_norm == 'ortho':  # Compute normalization factors.
      self._fft_norm_factor = tf.math.reciprocal(
          tf.math.sqrt(tf.cast(self._image_shape.num_elements(), dtype)))

    # Normalize coil sensitivities.
    self._sens_norm = sens_norm
    if self._sensitivities is not None and self._sens_norm:
      self._sensitivities = utils.normalize_no_nan(
          self._sensitivities, axis=-(self._rank + 1))

    # Set dynamic domain.
    if dynamic_domain is not None and self._extra_shape.rank == 0:
      raise ValueError(
          "Argument `dynamic_domain` requires a non-scalar `extra_shape`.")
    if dynamic_domain is not None:
      self._dynamic_domain = utils.validate_enum(
          dynamic_domain, {'time', 'frequency'}, name='dynamic_domain')
    else:
      self._dynamic_domain = None

    # This variable is used by `LinearOperatorGramMRI` to disable the NUFFT.
    self._skip_nufft = False

    super().__init__(dtype, name=name, parameters=parameters)


  
  def _transform(self, x, adjoint=False):
    """Transform [batch] input `x`.

    Args:
      x: A `tf.Tensor` of type `self.dtype` and shape
        `[..., *self.domain_shape]` containing images, if `adjoint` is `False`,
        or a `tf.Tensor` of type `self.dtype` and shape
        `[..., *self.range_shape]` containing *k*-space data, if `adjoint` is
        `True`.
      adjoint: A `boolean` indicating whether to apply the adjoint of the
        operator.

    Returns:
      A `tf.Tensor` of type `self.dtype` and shape `[..., *self.range_shape]`
      containing *k*-space data, if `adjoint` is `False`, or a `tf.Tensor` of
      type `self.dtype` and shape `[..., *self.domain_shape]` containing
      images, if `adjoint` is `True`.

    """
    
    if adjoint:
      # Apply density compensation.
      if self._density is not None and not self._skip_nufft:
        x *= self._dens_weights_sqrt

      # Apply adjoint Fourier operator.
      if self.is_non_cartesian:  # Non-Cartesian imaging, use NUFFT.
        if not self._skip_nufft:
          
          nufft_ob = KbNufftModule(im_size=(self._image_shape[0],self._image_shape[0]), grid_size=(self._image_shape[0]*2, self._image_shape[0]*2), norm='ortho')
          interpob = nufft_ob._extract_nufft_interpob()
          nufft_adj = kbnufft_adjoint(interpob)
          if len(self._trajectory.shape) == 4:
            x = nufft_adj(x, tf.transpose(self._trajectory[:,0,:,:], perm=[0,2,1]))
          else :  

            x = nufft_adj(x, tf.transpose(self._trajectory, perm=[0,2,1]))
          if self._fft_norm is not None:
            x *= self._fft_norm_factor

      else:  # Cartesian imaging, use FFT.
        if self._mask is not None:
          x *= self._mask_linop_dtype  # Undersampling.
        x = fft.ifftn(x, axes=self._image_axes,
                          norm=self._fft_norm or 'forward', shift=True)

      # Apply coil combination.
      if self.is_multicoil:
        x *= tf.math.conj(self._sensitivities)
        x = tf.math.reduce_sum(x, axis=-(self._rank + 1))

      # Maybe remove phase from image.
      if self.is_phase_constrained:
        x *= tf.math.conj(self._phase_rotator)
        x = tf.cast(tf.math.real(x), self.dtype)

      # Apply FFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft.fftn(x, axes=[self.dynamic_axis],
                         norm='ortho', shift=True)

    else:  # Forward operator.

      # Apply FFT along dynamic axis, if necessary.
      if self.is_dynamic and self.dynamic_domain == 'frequency':
        x = fft.ifftn(x, axes=[self.dynamic_axis],
                          norm='ortho', shift=True)

      # Add phase to real-valued image if reconstruction is phase-constrained.
      if self.is_phase_constrained:
        x = tf.cast(tf.math.real(x), self.dtype)
        x *= self._phase_rotator

      # Apply sensitivity modulation.
      if self.is_multicoil:
        x = tf.expand_dims(x, axis=-(self._rank + 1))
        x *= self._sensitivities

      # Apply Fourier operator.
      if self.is_non_cartesian:  # Non-Cartesian imaging, use NUFFT.
        if not self._skip_nufft:
          nufft_ob = KbNufftModule(im_size=(self._image_shape[0],self._image_shape[0]), grid_size=(self._image_shape[0]*2, self._image_shape[0]*2), norm='ortho')
          x = kbnufft_forward(nufft_ob._extract_nufft_interpob())(x, tf.transpose(self._trajectory[:,0,:,:], perm=[0, 2, 1]))
          
          if self._fft_norm is not None:
            x *= self._fft_norm_factor

      else:  # Cartesian imaging, use FFT.
        x = fft.fftn(x, axes=self._image_axes,
                         norm=self._fft_norm or 'backward', shift=True)
        if self._mask is not None:
          x *= self._mask_linop_dtype  # Undersampling.

      # Apply density compensation.
      if self._density is not None and not self._skip_nufft:
        x *= self._dens_weights_sqrt

    return x

  def _domain_shape(self):
    """Returns the shape of the domain space of this operator."""
    return self._extra_shape.concatenate(self._image_shape)

  def _range_shape(self):
    """Returns the shape of the range space of this operator."""
    if self.is_cartesian:
      range_shape = self._image_shape.as_list()
    else:
      range_shape = [self._trajectory.shape[-2]]
    if self.is_multicoil:
      range_shape = [self.num_coils] + range_shape
    return self._extra_shape.concatenate(range_shape)

  def _batch_shape(self):
    """Returns the static batch shape of this operator."""
    return self._batch_shape_value[:-self._extra_shape.rank or None]  # pylint: disable=invalid-unary-operand-type

  def _batch_shape_tensor(self):
    """Returns the dynamic batch shape of this operator."""
    return self._batch_shape_tensor_value[:-self._extra_shape.rank or None]  # pylint: disable=invalid-unary-operand-type

  @property
  def image_shape(self):
    """The image shape."""
    return self._image_shape

  @property
  def rank(self):
    """The number of spatial dimensions."""
    return self._rank

  @property
  def is_cartesian(self):
    """Whether this is a Cartesian MRI operator."""
    return self._trajectory is None

  @property
  def is_non_cartesian(self):
    """Whether this is a non-Cartesian MRI operator."""
    return self._trajectory is not None

  @property
  def is_multicoil(self):
    """Whether this is a multicoil MRI operator."""
    return self._sensitivities is not None

  @property
  def is_phase_constrained(self):
    """Whether this is a phase-constrained MRI operator."""
    return self._phase is not None

  @property
  def is_dynamic(self):
    """Whether this is a dynamic MRI operator."""
    return self._dynamic_domain is not None

  @property
  def dynamic_domain(self):
    """The dynamic domain of this operator."""
    return self._dynamic_domain

  @property
  def dynamic_axis(self):
    """The dynamic axis of this operator."""
    return -(self._rank + 1) if self.is_dynamic else None

  @property
  def num_coils(self):
    """The number of coils."""
    if self._sensitivities is None:
      return None
    return self._sensitivities.shape[-(self._rank + 1)]

  @property
  def _composite_tensor_fields(self):
    return ("image_shape", "mask", "trajectory", "density", "sensitivities",
            "fft_norm")
  

class LinearOperatorGramMatrix(LinearOperator):  # pylint: disable=abstract-method
  r"""Linear operator representing the Gram matrix of an operator.

  If :math:`A` is a `LinearOperator`, this operator is equivalent to
  :math:`A^H A`.

  The Gram matrix of :math:`A` appears in the normal equation
  :math:`A^H A x = A^H b` associated with the least squares problem
  :math:`{\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2}`.

  This operator is self-adjoint and positive definite. Therefore, linear systems
  defined by this linear operator can be solved using the conjugate gradient
  method.

  This operator supports the optional addition of a regularization parameter
  :math:`\lambda` and a transform matrix :math:`T`. If these are provided,
  this operator becomes :math:`A^H A + \lambda T^H T`. This appears
  in the regularized normal equation
  :math:`\left ( A^H A + \lambda T^H T \right ) x = A^H b + \lambda T^H T x_0`,
  associated with the regularized least squares problem
  :math:`{\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2 + \lambda \left \| T(x-x_0) \right \|_2^2}`.

  Args:
    operator: A `tfmri.linalg.LinearOperator`. The operator :math:`A` whose Gram
      matrix is represented by this linear operator.
    reg_parameter: A `Tensor` of shape `[B1, ..., Bb]` and real dtype.
      The regularization parameter :math:`\lambda`. Defaults to 0.
    reg_operator: A `tfmri.linalg.LinearOperator`. The regularization transform
      :math:`T`. Defaults to the identity.
    gram_operator: A `tfmri.linalg.LinearOperator`. The Gram matrix
      :math:`A^H A`. This may be optionally provided to use a specialized
      Gram matrix implementation. Defaults to `None`.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.
  """
  def __init__(self,
               operator,
               reg_parameter=None,
               reg_operator=None,
               gram_operator=None,
               is_non_singular=None,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               name=None):
    parameters = dict(
        operator=operator,
        reg_parameter=reg_parameter,
        reg_operator=reg_operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)
    self._operator = operator
    self._reg_parameter = reg_parameter
    self._reg_operator = reg_operator
    self._gram_operator = gram_operator
    if gram_operator is not None:
      self._composed = gram_operator
    else:
      self._composed = LinearOperatorComposition(
          operators=[self._operator.H, self._operator])

    if not is_self_adjoint:
      raise ValueError("A Gram matrix is always self-adjoint.")
    if not is_positive_definite:
      raise ValueError("A Gram matrix is always positive-definite.")
    if not is_square:
      raise ValueError("A Gram matrix is always square.")

    if self._reg_parameter is not None:
      reg_operator_gm = LinearOperatorScaledIdentity(
          shape=self._operator.domain_shape,
          multiplier=tf.cast(self._reg_parameter, self._operator.dtype))
      if self._reg_operator is not None:
        reg_operator_gm = LinearOperatorComposition(
            operators=[reg_operator_gm,
                       self._reg_operator.H,
                       self._reg_operator])
      self._composed = LinearOperatorAddition(
          operators=[self._composed, reg_operator_gm])

    super().__init__(operator.dtype,
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     parameters=parameters)
  def _transform(self, x, adjoint=False):
    return self._composed.transform(x, adjoint=adjoint)

  def _domain_shape(self):
    return self.operator.domain_shape

  def _range_shape(self):
    return self.operator.domain_shape

  def _batch_shape(self):
    return self.operator.batch_shape

  def _domain_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()

  @property
  def operator(self):
    return self._operator


class LinearOperatorComposition(LinalgImagingMixin,  # pylint: disable=abstract-method
                                tf.linalg.LinearOperatorComposition):
  """Composes one or more linear operators.

  `LinearOperatorComposition` is initialized with a list of operators
  :math:`A_1, A_2, ..., A_J` and represents their composition
  :math:`A_1 A_2 ... A_J`.

  .. note:
    Similar to `tf.linalg.LinearOperatorComposition`_, but with imaging
    extensions.

  Args:
    operators: A `list` of `LinearOperator` objects, each with the same `dtype`
      and composable shape.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_o_`.

  .. _tf.linalg.LinearOperatorComposition: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorComposition
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    if adjoint:
      transform_order_list = self.operators
    else:
      transform_order_list = list(reversed(self.operators))

    result = transform_order_list[0]._transform(x, adjoint=adjoint)
    for operator in transform_order_list[1:]:
      result = operator._transform(result, adjoint=adjoint)
    return result

  def _domain_shape(self):
    return self.operators[-1].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return utils.broadcast_static_shapes(
        *[operator.batch_shape for operator in self.operators])

  def _domain_shape_tensor(self):
    return self.operators[-1].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return utils.broadcast_dynamic_shapes(
        *[operator.batch_shape_tensor() for operator in self.operators])

class LinearOperatorAddition(LinalgImagingMixin,  # pylint: disable=abstract-method
                             LinearOperatorAddition):
  """Adds one or more linear operators.

  `LinearOperatorAddition` is initialized with a list of operators
  :math:`A_1, A_2, ..., A_J` and represents their addition
  :math:`A_1 + A_2 + ... + A_J`.

  Args:
    operators: A `list` of `LinearOperator` objects, each with the same `dtype`
      and shape.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_p_`.
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    result = self.operators[0]._transform(x, adjoint=adjoint)
    for operator in self.operators[1:]:
      result += operator._transform(x, adjoint=adjoint)
    return result

  def _domain_shape(self):
    return self.operators[0].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return  utils.broadcast_static_shapes(
        *[operator.batch_shape for operator in self.operators])

  def _domain_shape_tensor(self):
    return self.operators[0].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return  utils.broadcast_dynamic_shapes(
        *[operator.batch_shape_tensor() for operator in self.operators])


class LinearOperatorScaledIdentity(LinalgImagingMixin,  # pylint: disable=abstract-method
                                   tf.linalg.LinearOperatorScaledIdentity):
  """Linear operator representing a scaled identity matrix.

  .. note:
    Similar to `tf.linalg.LinearOperatorScaledIdentity`_, but with imaging
    extensions.

  Args:
    shape: Non-negative integer `Tensor`. The shape of the operator.
    multiplier: A `Tensor` of shape `[B1, ..., Bb]`, or `[]` (a scalar).
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form `x^H A x` has positive real part for all
      nonzero `x`.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.  See:
      https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
    is_square:  Expect that this operator acts like square [batch] matrices.
    assert_proper_shapes: Python `bool`.  If `False`, only perform static
      checks that initialization and method arguments have proper shape.
      If `True`, and static checks are inconclusive, add asserts to the graph.
    name: A name for this `LinearOperator`.

  .. _tf.linalg.LinearOperatorScaledIdentity: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorScaledIdentity
  """
  def __init__(self,
               shape,
               multiplier,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=True,
               assert_proper_shapes=False,
               name="LinearOperatorScaledIdentity"):

    self._domain_shape_tensor_value =  utils.convert_shape_to_tensor(
        shape, name="shape")
    self._domain_shape_value = tf.TensorShape(tf.get_static_value(
        self._domain_shape_tensor_value))

    super().__init__(
        num_rows=tf.math.reduce_prod(shape),
        multiplier=multiplier,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        assert_proper_shapes=assert_proper_shapes,
        name=name)

  def _transform(self, x, adjoint=False):
    domain_rank = tf.size(self.domain_shape_tensor())
    multiplier_shape = tf.concat([
        tf.shape(self.multiplier),
        tf.ones((domain_rank,), dtype=tf.int32)], 0)
    multiplier_matrix = tf.reshape(self.multiplier, multiplier_shape)
    if adjoint:
      multiplier_matrix = tf.math.conj(multiplier_matrix)
    return x * multiplier_matrix

  def _domain_shape(self):
    return self._domain_shape_value

  def _range_shape(self):
    return self._domain_shape_value

  def _batch_shape(self):
    return self.multiplier.shape

  def _domain_shape_tensor(self):
    return self._domain_shape_tensor_value

  def _range_shape_tensor(self):
    return self._domain_shape_tensor_value

  def _batch_shape_tensor(self):
    return tf.shape(self.multiplier)

AdmmOptimizerResults = collections.namedtuple(
    'AdmmOptimizerResults', [
        'converged',
        'dual_residual',
        'dual_tolerance',
        'f_primal_variable',
        'g_primal_variable',
        'num_iterations',
        'primal_residual',
        'primal_tolerance',
        'scaled_dual_variable'
    ]
)


def conjugate_gradient(operator,
                       rhs,
                       preconditioner=None,
                       x=None,
                       tol=1e-5,
                       max_iterations=20,
                       bypass_gradient=False,
                       name=None):
  

  return _conjugate_gradient_internal(operator, rhs,
                                      preconditioner=preconditioner,
                                      x=x,
                                      tol=tol,
                                      max_iterations=max_iterations,
                                      name=name)

def _conjugate_gradient_internal(operator,
                                 rhs,
                                 preconditioner=None,
                                 x=None,
                                 tol=1e-5,
                                 max_iterations=20,
                                 name=None):
  """Implementation of `conjugate_gradient`.

  For the parameters, see `conjugate_gradient`.
  """
  if isinstance(operator, LinalgImagingMixin):
    rhs = operator.flatten_domain_shape(rhs)

  if not (operator.is_self_adjoint and operator.is_positive_definite):
    raise ValueError('Expected a self-adjoint, positive definite operator.')

  cg_state = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

  def stopping_criterion(i, state):
    return tf.math.logical_and(
        i < max_iterations,
        tf.math.reduce_any(
            tf.math.real(tf.norm(state.r, axis=-1)) > tf.math.real(tol)))

  def dot(x, y):
    return tf.squeeze(
        tf.linalg.matvec(
            x[..., tf.newaxis],
            y, adjoint_a=True), axis=-1)

  def cg_step(i, state):  # pylint: disable=missing-docstring
    z = tf.linalg.matvec(operator, state.p)
    alpha = state.gamma / dot(state.p, z)
    x = state.x + alpha[..., tf.newaxis] * state.p
    r = state.r - alpha[..., tf.newaxis] * z
    if preconditioner is None:
      q = r
    else:
      q = preconditioner.matvec(r)
    gamma = dot(r, q)
    beta = gamma / state.gamma
    p = q + beta[..., tf.newaxis] * state.p
    return i + 1, cg_state(i + 1, x, r, p, gamma)

  # We now broadcast initial shapes so that we have fixed shapes per iteration.

  with tf.name_scope(name or 'conjugate_gradient'):
    broadcast_shape = tf.broadcast_dynamic_shape(
        tf.shape(rhs)[:-1],
        operator.batch_shape_tensor())
    static_broadcast_shape = tf.broadcast_static_shape(
        rhs.shape[:-1],
        operator.batch_shape)
    if preconditioner is not None:
      broadcast_shape = tf.broadcast_dynamic_shape(
          broadcast_shape,
          preconditioner.batch_shape_tensor())
      static_broadcast_shape = tf.broadcast_static_shape(
          static_broadcast_shape,
          preconditioner.batch_shape)
    broadcast_rhs_shape = tf.concat([broadcast_shape, [tf.shape(rhs)[-1]]], -1)
    static_broadcast_rhs_shape = static_broadcast_shape.concatenate(
        [rhs.shape[-1]])
    r0 = tf.broadcast_to(rhs, broadcast_rhs_shape)
    tol *= tf.norm(r0, axis=-1)

    if x is None:
      x = tf.zeros(
          broadcast_rhs_shape, dtype=rhs.dtype.base_dtype)
      x = tf.ensure_shape(x, static_broadcast_rhs_shape)
    else:
      r0 = rhs - tf.linalg.matvec(operator, x)
    if preconditioner is None:
      p0 = r0
    else:
      p0 = tf.linalg.matvec(preconditioner, r0)
    gamma0 = dot(r0, p0)
    i = tf.constant(0, dtype=tf.int32)
    state = cg_state(i=i, x=x, r=r0, p=p0, gamma=gamma0)
    _, state = tf.while_loop(
        stopping_criterion, cg_step, [i, state])

    if isinstance(operator, LinalgImagingMixin):
      x = operator.expand_range_dimension(state.x)
    else:
      x = state.x

    return cg_state(
        state.i,
        x=x,
        r=state.r,
        p=state.p,
        gamma=state.gamma)


def admm_minimize(function_f,
                  function_g,
                  operator_a=None,
                  operator_b=None,
                  constant_c=None,
                  penalty=1.0,
                  absolute_tolerance=1e-5,
                  relative_tolerance=1e-5,
                  max_iterations=50,
                  linearized=False,
                  f_prox_kwargs=None,
                  g_prox_kwargs=None,
                  name=None):
  r"""Applies the ADMM algorithm to minimize a separable convex function.

  Minimizes :math:`f(x) + g(z)`, subject to :math:`Ax + Bz = c`.

  If :math:`A`, :math:`B` and :math:`c` are not provided, the constraint
  defaults to :math:`x - z = 0`, in which case the problem is equivalent to
  minimizing :math:`f(x) + g(x)`.

  Args:
    function_f: A `tfmri.convex.ConvexFunction` of shape `[..., n]` and real or
      complex dtype.
    function_g: A `tfmri.convex.ConvexFunction` of shape `[..., m]` and real or
      complex dtype.
    operator_a: A `tf.linalg.LinearOperator` of shape `[..., p, n]` and real or
      complex dtype. Defaults to the identity operator.
    operator_b: A `tf.linalg.LinearOperator` of shape `[..., p, m]` and real or
      complex dtype. Defaults to the negated identity operator.
    constant_c: A `tf.Tensor` of shape `[..., p]`. Defaults to 0.0.
    penalty: A scalar `tf.Tensor`. The penalty parameter of the augmented
      Lagrangian. Also corresponds to the step size of the dual variable update
      in the scaled form of ADMM.
    absolute_tolerance: A scalar `tf.Tensor` of real dtype. The absolute
      tolerance. Defaults to 1e-5.
    relative_tolerance: A scalar `tf.Tensor` of real dtype. The relative
      tolerance. Defaults to 1e-5.
    max_iterations: A scalar `tf.Tensor` of integer dtype. The maximum number
      of iterations of the ADMM update.
    linearized: A `boolean`. If `True`, use linearized variant of the ADMM
      algorithm. Linearized ADMM solves problems of the form
      :math:`f(x) + g(Ax)` and only requires evaluation of the proximal operator
      of `g(x)`. This is useful when the proximal operator of `g(Ax)` cannot be
      easily evaluated, but the proximal operator of `g(x)` can. Defaults to
      `False`.
    f_prox_kwargs: A `dict`. Keyword arguments to pass to the proximal operator
      of `function_f` during the x-minimization step.
    g_prox_kwargs: A `dict`. Keyword arguments to pass to the proximal operator
      of `function_g` during the z-minimization step.
    name: A `str`. The name of this operation. Defaults to `'admm_minimize'`.

  Returns:
    A `namedtuple` containing the following fields

    - `converged`: A boolean `tf.Tensor` of shape `[...]` indicating whether the
      minimum was found within tolerance for each batch member.
    - `dual_residual`: A real `tf.Tensor` of shape `[...]` containing the
      last tolerance used to evaluate the primal feasibility condition.
    - `dual_tolerance`: The dual tolerance.
    - `f_primal_variable`: A real or complex `tf.Tensor` of shape `[..., n]`
      containing the last argument value of `f` found during the search for
      each batch member. If the search converged, then this value is the argmin
      of the objective function, subject to the specified constraint.
    - `g_primal_variable`: A real or complex `tf.Tensor` of shape `[..., m]`
      containing the last argument value of `g` found during the search for
      each batch member. If the search converged, then this value is the argmin
      of the objective function, subject to the specified constraint.
    - `num_iterations`: A scalar integer `tf.Tensor` containing the number of
      iterations of the ADMM update.
    - `primal_residual`: A real or complex `tf.Tensor` of shape `[..., p]`
      containing the last primal residual for each batch member.
    - `primal_tolerance`: A real `tf.Tensor` of shape `[...]` containing the
      last tolerance used to evaluate the primal feasibility condition.
    - `scaled_dual_variable`: A `tf.Tensor` of shape `[..., p]` and real or
      complex dtype containing the last value of the scaled dual variable found
      during the search.

  References:
    .. [1] Boyd, S., Parikh, N., & Chu, E. (2011). Distributed optimization and
      statistical learning via the alternating direction method of multipliers.
      Now Publishers Inc.

  Raises:
    TypeError: If inputs have incompatible types.
    ValueError: If inputs are incompatible.
  """
  with tf.name_scope(name or 'admm_minimize'):
    if linearized:
      if operator_b is not None:
        raise ValueError(
            "Linearized ADMM does not support the use of `operator_b`.")
      if constant_c is not None:
        raise ValueError(
            "Linearized ADMM does not support the use of `constant_c`.")

    # Infer the dtype of the variables from the dtype of f.
    dtype = tf.dtypes.as_dtype(function_f.dtype)

    # Check that dtypes of both functions match.
    if function_g.dtype != dtype:
      raise TypeError(
          f"`function_f` and `function_g` must have the same dtype, but "
          f"got: {dtype} and {function_g.dtype}")

    # Check that batch shapes of both functions match.
    batch_shape = utils.batch_shape(function_f)
    batch_shape = utils.broadcast_shape(
        batch_shape, function_g.batch_shape_tensor())

    # Infer the dimensionality of the primal variables x, z from the
    # dimensionality of the domains of f and g.
    x_ndim_static = function_f.domain_dimension
    z_ndim_static = function_g.domain_dimension
    x_ndim = utils.domain_dimension(function_f)
    z_ndim = utils.domain_dimension(function_g)

    # Provide default values for A and B.
    if operator_a is None:
      operator_a = tf.linalg.LinearOperatorScaledIdentity(
          x_ndim, tf.constant(1.0, dtype=dtype))
    if operator_b is None:
      operator_b = tf.linalg.LinearOperatorScaledIdentity(
          z_ndim, tf.constant(-1.0, dtype=dtype))

    # Statically check that the domain shapes of the A, B operators are
    # consistent with f and g.
    if not operator_a.shape[-1:].is_compatible_with([x_ndim_static]):
      raise ValueError(
          f"`operator_a` must have the same domain dimension as `function_f`, "
          f"but got: {operator_a.shape[-1]} and {x_ndim_static}")
    if not operator_b.shape[-1:].is_compatible_with([z_ndim_static]):
      raise ValueError(
          f"`operator_b` must have the same domain dimension as `function_g`, "
          f"but got: {operator_b.shape[-1]} and {z_ndim_static}")

    # Check the batch shapes of the operators.
    batch_shape = utils.broadcast_shape(
        batch_shape, operator_a.batch_shape_tensor())
    batch_shape = utils.broadcast_shape(
        batch_shape, operator_b.batch_shape_tensor())

    # Infer the dimensionality of the dual variable u from the range shape of
    # operator A.
    u_ndim_static = operator_a.range_dimension
    if isinstance(u_ndim_static, tf.compat.v1.Dimension):
      u_ndim_static = u_ndim_static.value
    u_ndim = utils.range_dimension(operator_a)

    # Check that the range dimension of operator B is compatible with that of
    # operator A.
    if not operator_b.shape[-2:-1].is_compatible_with([u_ndim_static]):
      raise ValueError(
          f"`operator_b` must have the same range dimension as `operator_a`, "
          f"but got: {operator_b.shape[-2]} and {u_ndim_static}")

    # Provide default value for constant c.
    if constant_c is None:
      constant_c = tf.constant(0.0, dtype=dtype, shape=[u_ndim])

    # Check that the constant c has the same dimensionality as the dual
    # variable.
    if not constant_c.shape[-1:].is_compatible_with([u_ndim_static]):
      raise ValueError(
          f"The last dimension of `constant_c` must be equal to the range "
          f"dimension of `operator_a`, but got: {constant_c.shape[-1]} and "
          f"{u_ndim_static}")

    if linearized:
      f_update_fn = function_f.prox
      g_update_fn = function_g.prox
    else:
      f_update_fn = _get_admm_update_fn(function_f, operator_a,
                                        prox_kwargs=f_prox_kwargs)
      g_update_fn = _get_admm_update_fn(function_g, operator_b,
                                        prox_kwargs=g_prox_kwargs)

    x_ndim_sqrt = tf.math.sqrt(tf.cast(x_ndim, dtype.real_dtype))
    u_ndim_sqrt = tf.math.sqrt(tf.cast(u_ndim, dtype.real_dtype))

    absolute_tolerance = tf.convert_to_tensor(
        absolute_tolerance, dtype=dtype.real_dtype, name='absolute_tolerance')
    relative_tolerance = tf.convert_to_tensor(
        relative_tolerance, dtype=dtype.real_dtype, name='relative_tolerance')
    max_iterations = tf.convert_to_tensor(
        max_iterations, dtype=tf.dtypes.int32, name='max_iterations')

    def _cond(state):
      """Returns `True` if optimization should continue."""
      return tf.math.logical_and(
          state.num_iterations < max_iterations,
          tf.math.reduce_any(tf.math.logical_not(state.converged)))

    def _body(state):  # pylint: disable=missing-param-doc
      """The ADMM update."""
      # x-minimization step.
      state_bz = tf.linalg.matvec(operator_b, state.g_primal_variable)
      if linearized:
        v = state.f_primal_variable - tf.linalg.matvec(
            operator_a,
            (tf.linalg.matvec(operator_a, state.f_primal_variable) -
             state.g_primal_variable + state.scaled_dual_variable),
            adjoint_a=True)
      else:
        v = constant_c - state_bz - state.scaled_dual_variable
      f_primal_variable = f_update_fn(v, penalty)

      # z-minimization step.
      ax = tf.linalg.matvec(operator_a, f_primal_variable)
      if linearized:
        v = ax + state.scaled_dual_variable
      else:
        v = constant_c - ax - state.scaled_dual_variable
      g_primal_variable = g_update_fn(v, penalty)

      # Dual variable update and compute residuals.
      bz = tf.linalg.matvec(operator_b, g_primal_variable)
      primal_residual = ax + bz - constant_c
      scaled_dual_variable = state.scaled_dual_variable + primal_residual
      dual_residual = penalty * tf.linalg.matvec(
          operator_a, bz - state_bz, adjoint_a=True)

      # Choose the primal tolerance.
      ax_norm = tf.math.real(tf.norm(ax, axis=-1))
      bz_norm = tf.math.real(tf.norm(bz, axis=-1))
      c_norm = tf.math.real(tf.norm(constant_c, axis=-1))
      max_norm = tf.math.maximum(tf.math.maximum(ax_norm, bz_norm), c_norm)
      primal_tolerance = (absolute_tolerance * u_ndim_sqrt +
                          relative_tolerance * max_norm)

      # Choose the dual tolerance.
      aty_norm = tf.math.real(tf.norm(
          tf.linalg.matvec(operator_a, penalty * state.scaled_dual_variable,
                           adjoint_a=True),
          axis=-1))
      dual_tolerance = (absolute_tolerance * x_ndim_sqrt +
                        relative_tolerance * aty_norm)

      # Check convergence.
      converged = tf.math.logical_and(
          tf.math.real(tf.norm(primal_residual, axis=-1)) <= primal_tolerance,
          tf.math.real(tf.norm(dual_residual, axis=-1)) <= dual_tolerance)

      return [AdmmOptimizerResults(converged=converged,
                                   dual_residual=dual_residual,
                                   dual_tolerance=dual_tolerance,
                                   f_primal_variable=f_primal_variable,
                                   g_primal_variable=g_primal_variable,
                                   num_iterations=state.num_iterations + 1,
                                   primal_residual=primal_residual,
                                   primal_tolerance=primal_tolerance,
                                   scaled_dual_variable=scaled_dual_variable)]
    # Initial state.
    x_shape = utils.concat([batch_shape, [x_ndim]], axis=0)
    z_shape = utils.concat([batch_shape, [z_ndim]], axis=0)
    u_shape = utils.concat([batch_shape, [u_ndim]], axis=0)

    state = AdmmOptimizerResults(
        converged=tf.fill(batch_shape, False),
        f_primal_variable=tf.zeros(shape=x_shape, dtype=dtype),
        g_primal_variable=tf.zeros(shape=z_shape, dtype=dtype),
        dual_residual=None,
        dual_tolerance=None,
        num_iterations=tf.constant(0, dtype=tf.dtypes.int32),
        primal_residual=None,
        primal_tolerance=None,
        scaled_dual_variable=tf.zeros(shape=u_shape, dtype=dtype))
    state = _body(state)[0]

    return tf.while_loop(_cond, _body, [state])[0]


def _get_admm_update_fn(function, operator, prox_kwargs=None):
  r"""Returns a function for the ADMM update.

  The returned function evaluates the expression
  :math:`{\mathop{\mathrm{argmin}}_x} \left ( f(x) + \frac{\rho}{2} \left\| Ax - v \right\|_2^2 \right )`
  for a given input :math:`v` and penalty parameter :math:`\rho`.

  This function will raise an error if the above expression cannot be easily
  evaluated for the specified convex function and linear operator.

  Args:
    function: A `ConvexFunction` instance.
    operator: A `LinearOperator` instance.
    prox_kwargs: A `dict` of keyword arguments to pass to the proximal operator
      of `function`.

  Returns:
    A function that evaluates the ADMM update.

  Raises:
    NotImplementedError: If no rules exist to evaluate the ADMM update for the
      specified inputs.
  """  # pylint: disable=line-too-long
  prox_kwargs = prox_kwargs or {}

  if isinstance(operator, tf.linalg.LinearOperatorIdentity):
    def _update_fn(x, rho):
      return function.prox(x, scale=1.0 / rho, **prox_kwargs)
    return _update_fn

  if isinstance(operator, tf.linalg.LinearOperatorScaledIdentity):
    # This is equivalent to multiplication by a scalar, which can be taken out
    # of the norm and pooled with `rho`. If multiplier is negative, we also
    # change the sign of `v` in order to obtain the expression of the proximal
    # operator of f.
    multiplier = operator.multiplier
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      return function.prox(
          tf.math.sign(multiplier) * v, scale=tf.math.abs(multiplier) / rho,
          **prox_kwargs)
    return _update_fn

  if isinstance(function, convex_ops.ConvexFunctionQuadratic):
    # TODO(jmontalt): add prox_kwargs here.
    # See ref. [1], section 4.2.
    def _update_fn(v, rho):  # pylint: disable=function-redefined
      solver_kwargs = prox_kwargs.get('solver_kwargs', {})
      # Create operator Q + rho * A^T A, where Q is the quadratic coefficient
      # of the quadratic convex function.
      scaled_identity = tf.linalg.LinearOperatorScaledIdentity(
          operator.shape[-1], tf.cast(rho, operator.dtype))
      ls_operator = tf.linalg.LinearOperatorComposition(
          [scaled_identity, operator.H, operator])
      ls_operator = LinearOperatorAddition(
          [function.quadratic_coefficient, ls_operator],
          is_self_adjoint=True, is_positive_definite=True)
      # Compute the right-hand side of the linear system.
      rhs = (rho * tf.linalg.matvec(operator, v, adjoint_a=True) -
             function.linear_coefficient)
      # Solve the linear system using CG (see ref [1], section 4.3.4).
      return conjugate_gradient(ls_operator, rhs, **solver_kwargs).x

    return _update_fn

  raise NotImplementedError(
      f"No rules to evaluate the ADMM update for function "
      f"{function.name} and operator {operator.name}.")

def lbfgs_minimize(*args, **kwargs):
  """Applies the L-BFGS algorithm to minimize a differentiable function.

  For the parameters, see `tfp.optimizer.lbfgs_minimize`_.

  .. _tfp.optimizer.lbfgs_minimize: https://www.tensorflow.org/probability/api_docs/python/tfp/optimizer/lbfgs_minimize
  """
  print('lbfgs minimization')
  return tfp.optimizer.lbfgs_minimize(*args, **kwargs)