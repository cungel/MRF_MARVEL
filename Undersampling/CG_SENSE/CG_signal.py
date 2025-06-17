import tensorflow as tf
import numpy as np
import CG_SENSE.utils as utils

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
  
def filter_kspace(kspace,
                trajectory=None,
                filter_fn='hann',
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
        kspace, trajectory = utils.verify_compatible_trajectory(
        kspace, trajectory)

    # Make a "trajectory" for Cartesian k-spaces.
    is_cartesian = trajectory is None
    if is_cartesian:
        filter_rank = filter_rank or kspace.shape.rank
        vecs = [tf.linspace(-np.pi, np.pi - (2.0 * np.pi / s), s)
                for s in kspace.shape[-filter_rank:]]  # pylint: disable=invalid-unary-operand-type
        trajectory = utils.meshgrid(*vecs)

    if not callable(filter_fn):
    # filter_fn not a callable, so should be an enum value. Get the
    # corresponding function.
        filter_fn = utils.validate_enum(
            filter_fn, valid_values={'hamming', 'hann', 'atanfilt'},
            name='filter_fn')
        filter_fn = {
            'hann': hann,
        }[filter_fn]
    filter_kwargs = filter_kwargs or {}

    traj_norm = tf.norm(trajectory, axis=-1)
    return kspace * tf.cast(filter_fn(traj_norm, **filter_kwargs), kspace.dtype)
