from gym.envs.registration import register

register(
    id='ao-v0',
    entry_point='ao_gym.envs:AOEnv',
)