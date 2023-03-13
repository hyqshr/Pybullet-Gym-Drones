from gym.envs.registration import register
register(
    id='DroneNavigation-v0',
    entry_point='drone_envs.envs:DroneNavigationV0'
)