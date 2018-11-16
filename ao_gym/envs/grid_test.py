"""Tests for grids.py"""

import numpy as np

from ao_gym.envs import grids

class gridTest(np.testing.TestCase):

  def testGrid(self):
    x = [[-1., 0., 1.]] * 3
    y = [[-1.] * 3, [0.] * 3, [1.] * 3]
    test_grid = grids.Grid((3, 3), (1., 1.))
    np.testing.assert_almost_equal(x, test_grid.x)
    np.testing.assert_almost_equal(y, test_grid.y)

  def testGridEvenSize(self):
    x = [[-.5, .5]] * 2
    y = [[-.5] * 2, [.5] * 2]
    test_grid = grids.Grid((2, 2), (1., 1.))
    np.testing.assert_almost_equal(x, test_grid.x)
    np.testing.assert_almost_equal(y, test_grid.y)

  def testGridnonequalPixels(self):
    x = [[-.6, -.3, 0., .3, .6]] * 3
    y = [[-1.] * 5, [0.] * 5, [1.] * 5]
    test_grid = grids.Grid((5, 3), (.3, 1.))
    np.testing.assert_almost_equal(x, test_grid.x)
    np.testing.assert_almost_equal(y, test_grid.y)

  def testPolarGrid(self):
    r = [[.5 * np.sqrt(2)] * 2 ] * 2
    theta = [[np.pi * - 3 / 4, np.pi * - 1 / 4],
             [np.pi * 3 / 4, np.pi * 1 / 4]]
    x = [[-.5, .5]] * 2
    y = [[-.5] * 2, [.5] * 2]
    test_grid = grids.Grid((2, 2), (1., 1.))
    test_r, test_theta = test_grid.polar()
    np.testing.assert_almost_equal(test_r, r)
    np.testing.assert_almost_equal(test_theta, theta)


  def testInvalidGridSizeX(self):
    with self.assertRaisesRegex(
        ValueError, "Logical size must be greater than 0 in all dimensions"):
      grids.Grid((0, 3), (.3, 1.))

  def testInvalidGridSizeY(self):
    with self.assertRaisesRegex(
        ValueError, "Logical size must be greater than 0 in all dimensions"):
      grids.Grid((2, 0), (.3, 1.))

  def testInvalidGridPitchX(self):
    with self.assertRaisesRegex(
        ValueError, "Pixel size must be greater than 0 in all dimensions"):
      grids.Grid((2, 3), (0, 1.))

  def testInvalidGridPitchY(self):
    with self.assertRaisesRegex(
        ValueError, "Pixel size must be greater than 0 in all dimensions"):
      grids.Grid((2, 3), (1., 0))