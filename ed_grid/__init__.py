from gym.envs.registration import register

register(
    id='ed-grid-v0',
    entry_point='ed_grid.envs:ED_Env',
)
