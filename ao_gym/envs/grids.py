"""Contains utilities for building and modifying grids."""

import numpy as np

from typing import Tuple

class Grid():
  """Contains a spatial grid in 2D.

  The `Grid` object represents space in (optionally multiple) discretizations.
  """

  x=None,
  y=None,

  def __init__(self, size, pixel_dimensions, indexing='xy'):
    if any(logical_dim<1 for logical_dim in size):
      raise ValueError("Logical size must be greater than 0 in all dimensions")

    if any(pixel_size<=0 for pixel_size in pixel_dimensions):
      raise ValueError("Pixel size must be greater than 0 in all dimensions")

    self.x, self.y = _grid(size, pixel_dimensions, indexing)

  def polar(self):
    return _cartesian_to_polar(self.x, self.y)

def _grid(
  size: Tuple[int, int],
  pixel_dimensions: Tuple[float, float],
  indexing: str = 'xy',
):
  """Generates a grid with values corresponding to physical location.

  This function can be used to compute x and y coordinates for a physical
  problem. The grids are indexed by default in an `xy` manner.

  Each array represents the value of a physical coordinate at the `center` of
  the represented grid element.

      0---> x
      |
      |
      y

  Explicitly:

    x, y = grid((5, 5), (.5, .5))

    # x = [[-1. , -0.5,  0. ,  0.5,  1. ],
    #      [-1. , -0.5,  0. ,  0.5,  1. ],
    #      [-1. , -0.5,  0. ,  0.5,  1. ],
    #      [-1. , -0.5,  0. ,  0.5,  1. ],
    #      [-1. , -0.5,  0. ,  0.5,  1. ]]

    # y = [[-2., -2., -2., -2., -2.],
    #      [-1., -1., -1., -1., -1.],
    #      [ 0.,  0.,  0.,  0.,  0.],
    #      [ 1.,  1.,  1.,  1.,  1.],
    #      [ 2.,  2.,  2.,  2.,  2.]]

  The origin of the grid is located at the center of the arrays. The spacing
  of grid values corresponds to `pixel_dimensions`.

  Args:
    size:
    pixel

  Returns:
    X: `tf.Tensor` describing physical x location of each grid position.
    Y: Same as `X` but for y-coordinates.

  Raises:
    ValueError: If inputs have incorrect shape or dtype.
  """
  if any(not isinstance(dimension, float) for dimension in pixel_dimensions):
    raise ValueError(
      "All values in `pixel_dimension` must be of float dtype, "
      "got {}".format(pixel_dimensions))

  x_maximum, y_maximum = ((size[0] - 1.) / 2 * pixel_dimensions[0],
                          (size[1] - 1.) / 2 * pixel_dimensions[1])

  x_coordinates = np.linspace(-x_maximum, x_maximum, size[0])
  y_coordinates = np.linspace(-y_maximum, y_maximum, size[1])

  return np.meshgrid(x_coordinates, y_coordinates, indexing=indexing)


def _cartesian_to_polar(x, y):
  """Generates polar coordinates from x and y coordinates.
  
  This function calculates the polar coordinates `(r, theta)` given a set of
  cartesian coordinates `(x, y)`.

  Args:
    x: Array of arbitrary shape describing x coordinates.
    y: Array of same shape as (or broadcast compatible with) x describing
      y-coordinates.

  Returns:
    r: `np.ndarray` of same shape as `x` representing radius coordinate.
    theta: Same as `r` but representing displacement angle.
  """
  # Calculate radius using `r ** 2 = x ** 2 + y ** 2`.
  r = np.sqrt(x ** 2 + y ** 2)

  # Calculate theta using `tan(theta) = y / x`.
  theta = np.arctan2(y, x)
  return r, theta