import gym
from gym import error, spaces, utils
from gym.utils import seeding

from ao_gym.envs import optics_simulation
import numpy as np
import tensorflow as tf

class AOEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # Set up random problem.
    self.graph = tf.Graph()

    # initial value is 32 random values.
    self.position = np.ones([32,32]) * np.random.rand([32, 1])
    self.dm_positions, self.focal_plane_intensity = optics_simulation.ao_model()

    self.session = tf.Session(graph=self.graph)

    self.current_image = self.session.eval(self.focal_plane_intensity)

    # Keep track of number of steps.  We want the algorithm to converge quickly.
    self.current_step = 1


  def _step(self, action):
    """

    Parameters
    ----------
    action :

    Returns
    -------
    ob, reward, episode_over, info : tuple
        ob (object) :
            an environment-specific object representing your observation of
            the environment.
        reward (float) :
            amount of reward achieved by the previous action. The scale
            varies between environments, but the goal is always to increase
            your total reward.
        episode_over (bool) :
            whether it's time to reset the environment again. Most (but not
            all) tasks are divided up into well-defined episodes, and done
            being True indicates the episode has terminated. (For example,
            perhaps the pole tipped too far, or you lost your last life.)
        info (dict) :
             diagnostic information useful for debugging. It can sometimes
             be useful for learning (for example, it might contain the raw
             probabilities behind the environment's last state change).
             However, official evaluations of your agent are not allowed to
             use this for learning.
    """
    print(self.current_step)

    # Action.
    self._take_action(action)

    # Update state.
    self._take_image()

    reward = self._get_reward()
    ob = self._get_state()
    episode_over = self._is_over()

    self.current_step += 1
    return ob, reward, episode_over, {}

  def _reset(self):
    # initial value is 32 random values.
    self.position = np.ones([32,32]) * np.random.rand([32, 1])
    self.current_step = 0

  def _render(self, mode='human', close=False):
    return

  def _take_action(self, action):
    self.position = self.position + action


  def _take_image(self):
    """Update current image."""
    self.current_image = self.session.eval(self.focal_plane_intensity)


  def _get_state(self):
    """Get the intensity in central area."""
    return self.current_image

  def _is_over(self):
    # End if has taken 200 steps.
    if self.current_step > 200:
      return True
    else:
      return False

  def _get_reward(self):
    """ Reward is given for intensity in center pixels. """
    return np.sum(self.current_image[0, ])
