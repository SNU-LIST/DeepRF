from gym.envs.registration import register

# Slice-selective excitation pulse
register(
    id='Exc-v51',
    entry_point='envs.deeprf.environment:DeepRFSLREXC20',
    kwargs={'sar_coef': 0.00001,
            'ripple_coef': 1.0,
            'sampling_rate': 256,
            'max_mag': 0.92724,
            'max_ripple': 0.0146,
            'pos_range1': (-1285, 1285, 1000),
            'pos_range2': (-32000, -1614, 1500),
            'pos_range3': (1614, 32000, 1500)}
)
