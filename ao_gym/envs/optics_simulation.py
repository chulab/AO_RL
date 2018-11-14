"""Defines basic optics simulation."""

from typing import Tuple

import numpy as np
import tensorflow as tf


def grid(
    size: Tuple[int, int],
    pixel_dimensions: Tuple[float, float],
    indexing: str = 'xy',
):
  """Generates a grid with values corresponding to physical location.

  This function can be used to compute x and y coordinates for a physical
  problem. The grids are indexed by default in an `xy` manner.

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
    raise ValueError("All values in `pixel_dimension` must be of float dtype, "
                     "got {}".format(pixel_dimensions))

  x_maximum, y_maximum = (size[0] // 2 * pixel_dimensions[0],
                          size[1] // 2 * pixel_dimensions[1])

  x_coordinates = np.linspace(-x_maximum, x_maximum, size[0])
  y_coordinates = np.linspace(-y_maximum, y_maximum, size[1])

  return np.meshgrid(x_coordinates, y_coordinates, indexing=indexing)


def fftshift(array):
  """Shifts 0 frequency element to center of array."""
  array.shape.assert_is_compatible_with([None] * 3)
  center = [dimension // 2 for dimension in array.shape.as_list()][1:]
  print(center)
  return tf.concat([tf.concat(
    [array[:, center[0]:, center[1]:], array[:, center[0]:, :center[1]]], -1),
                    tf.concat([array[:, :center[0], center[1]:],
                               array[:, :center[0], :center[1]]], -1)], -2)


def intensity(input_field):
  """Computes intensity (image) of field."""
  return tf.abs(input_field)


def thin_film_focal(
    input_field: tf.Tensor
):
  """Computes field at focal plane of thin lens.

  See `http://www.photonics.intec.ugent.be/download/ocs130.pdf`, Equation 4.120.


  Args:
    input_field: `tf.Tensor` of shape `batch_dimensions + [height, width]`.

  Returns:
    `tf.Tensor` representing field at focal plane.

  Raises:
    ValueError: If arguments have incorrect dtype.
  """
  if input_field.dtype not in (tf.complex64, tf.complex128):
    raise ValueError("`input_field` must be complex but got {}".format(
      input_field.dtype))

  fourier_field = tf.fft2d(input_field)
  fourier_field = fftshift(fourier_field)

  return fourier_field


def image_to_pupil_units(focal_length, wavelength, grid_units, sample_count):
  """Computes pupil plane grid units."""
  return tuple([focal_length * wavelength / (grid * count) for
                grid, count in zip(grid_units, sample_count)])


def rescale_grid(
    grids: Tuple,
    current_units,
    destination_units,
):
  """Rescales unit of grid from `current_units` to `destination_units`."""
  if len(grids) != 2 or grids[0].shape != grids[1].shape:
    raise ValueError(
      "`grids` must contain two arrays corresponding to the same"
      "plane, but got {}".format(grids))

  return [grid / current * destination for grid, current, destination in
          zip(grids, current_units, destination_units)]


def nearest_neighbor_interpolation(
    train_points,
    train_values,
    query_points,
):
  """Performs nearest neighbor interpolation.

  Args:
    train_points

  """

  displacement_vectors = query_points[:, :, tf.newaxis, :] - train_points[:,
                                                             tf.newaxis, :, :]

  # Comput distance between all `train_points` and `query_points`.
  # `displacement_length` has shape `[batch, query_count, train_count]`.
  displacement_length = tf.reduce_sum(displacement_vectors ** 2, -1)

  # Find indices with minimum distance along `train_count` axis.
  # `query_indices` is a tensor of indices into `train_values` with shape
  # `[batch, query_count]`
  query_indices = tf.math.argmin(displacement_length, axis=-1)

  print(train_values)
  print(query_indices)

  return tf.batch_gather(train_values, tf.cast(query_indices, tf.int32))


def grid_dimension(
    grids,
):
  """Returns X and Y dimensions of `grids`."""
  return grids[0][0, -1] - grids[0][0, 0], grids[1][-1, 0] - grids[1][0, 0]


def grid_pitch(
    grids,
):
  """Returns X and Y grid pitch."""
  return grids[0][0, 1] - grids[0][0, 0], grids[1][1, 0] - grids[1][0, 0]


def deformable_mirror(
    dm_positions,
    pitch_size,
    pupil_grids,
):
  """Generates transfer function for deformable mirror evaluated on `grid`."""

  grid_dimensions = grid_dimension(pupil_grids)

  pupil_grid_pitch = grid_pitch(pupil_grids)

  dm_dimension = [pitch * size for pitch, size in
                  zip(pitch_size, dm_positions.shape.as_list()[-2:])]

  dm_size_in_pupil_dimension = [int(dimension / pitch) for dimension, pitch in
                                zip(dm_dimension, pupil_grid_pitch)]

  top_pad = [int((pupil_size - dm_size) // 2) for pupil_size, dm_size in
             zip(pupil_grids[0].shape[-2:], dm_size_in_pupil_dimension)]
  bottom_pad = [pupil_size - dm_size - top_pad_size for
                pupil_size, dm_size, top_pad_size in
                zip(pupil_grids[0].shape[-2:], dm_size_in_pupil_dimension,
                    top_pad)]

  dm_positions = tf.image.resize_images(
    dm_positions[..., tf.newaxis],
    dm_size_in_pupil_dimension,
    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

  # Pad with zeros so values outside of DM get no phase shift.
  dm_positions = tf.pad(dm_positions, [[0, 0], [top_pad[0], bottom_pad[0]],
                                       [top_pad[1], bottom_pad[1]]],
                        mode="constant")

  return dm_positions


def amplitude_phase_to_field(
    amplitude: tf.Tensor,
    phase: tf.Tensor,
):
  """Builds a complex field from amplitude and phase values.

  Args:
    phase: `tf.Tensor` representing phase.

  Returns:
    `tf.Tensor` containing complex field.
  """
  if any(tensor.dtype.is_complex for tensor in [amplitude, phase]):
    raise ValueError(
      "`amplitude` and `phase` tensors should be real valued, but"
      "got {}".format([amplitude.dtype, phase.dtype]))
  return (tf.cast(amplitude, tf.complex64) *
          tf.exp(-1j * tf.cast(phase, tf.complex64)))


def make_aperture(
    aperture_size,
    grids,
):
  """Makes complex transmission tensor for aperture."""
  aperture_amplitude = (np.sqrt(grids[0] ** 2 + grids[1] ** 2)
                        < aperture_size)
  aperture_phase = np.zeros_like(grids[0])
  return amplitude_phase_to_field(tf.convert_to_tensor(aperture_amplitude),
                                 tf.convert_to_tensor(aperture_phase))


def ao_model():
  """Builds tensorflow graph to compute optics calculations.

  Returns:
    dm_positions: `tf.placeholder` to feed in dm positions.
    focal_plane_intensity: `tf.Tensor` containing intensity at sensor.
  """
  pixel_pitch = (2.5, 2.5)  # um.
  grid_size = (300, 300)
  focal_length = 10e4  # um.
  wavelength = .5  # um.

  image_grids = grid(grid_size, pixel_pitch)

  # Convert to pupil plane.
  pupil_grid_units = image_to_pupil_units(focal_length, wavelength,
                                          pixel_pitch, grid_size)
  pupil_grids = rescale_grid(image_grids, pixel_pitch, pupil_grid_units)

  # Deformable mirror.
  # Specs from
  # http://www.bostonmicromachines.com/mid-actuator-count.html `Kilo-S-SLM`

  dm_pitch = (300., 300.)  # um.
  actuators = (32, 32)
  pupil_grids = pupil_grids

  dm_positions = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32])

  dm_physical_positions = deformable_mirror(dm_positions, dm_pitch,
                                            pupil_grids)

  # DM
  dm_phase = dm_physical_positions * np.pi * 2
  dm_amplitude = tf.ones_like(dm_phase)
  dm_field = amplitude_phase_to_field(dm_amplitude, dm_phase)

  # Aperture
  aperture_size = .4e4  # um.
  aperture = make_aperture(aperture_size, pupil_grids)

  # Combined.
  field = dm_field * aperture[tf.newaxis, ...]

  # Imaging.
  focal_plane_field = thin_film_focal(field)
  focal_plane_intensity = intensity(focal_plane_field)

  return dm_positions, focal_plane_intensity

