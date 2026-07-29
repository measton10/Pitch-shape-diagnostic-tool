"""
Generates simulated shape data for 5 fictional "college pitchers,"
standing in for a real college pitching staff since real TrackMan
data wasn't available for this project. Shape ranges are based on
realistic MLB/college-level values.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

PITCH_PROFILES = {
    'FF': {'speed': (88, 96), 'spin': (2000, 2400), 'axis': (180, 220), 'pfx_x': (-0.05, 0.15), 'pfx_z': (0.10, 0.20), 'ext': (5.5, 6.5)},
    'SL': {'speed': (78, 85), 'spin': (2200, 2700), 'axis': (60, 100), 'pfx_x': (-0.10, 0.10), 'pfx_z': (-0.05, 0.05), 'ext': (5.5, 6.5)},
    'CH': {'speed': (80, 87), 'spin': (1500, 1900), 'axis': (200, 240), 'pfx_x': (-0.10, 0.15), 'pfx_z': (0.00, 0.10), 'ext': (5.5, 6.5)},
    'CU': {'speed': (72, 79), 'spin': (2400, 2900), 'axis': (30, 70), 'pfx_x': (-0.15, 0.05), 'pfx_z': (-0.15, -0.05), 'ext': (5.3, 6.3)},
}

rows = []
for pitcher_num in range(1, 6):
    pitcher_name = f"Sample Pitcher {pitcher_num}"
    for pitch_type, ranges in PITCH_PROFILES.items():
        for _ in range(100):
            rows.append({
                'player_name': pitcher_name,
                'pitch_type': pitch_type,
                'release_speed': np.random.uniform(*ranges['speed']),
                'release_spin_rate': np.random.uniform(*ranges['spin']),
                'spin_axis': np.random.uniform(*ranges['axis']),
                'pfx_x': np.random.uniform(*ranges['pfx_x']),
                'pfx_z': np.random.uniform(*ranges['pfx_z']),
                'release_extension': np.random.uniform(*ranges['ext']),
            })

sample_df = pd.DataFrame(rows)
sample_df.to_csv('sample_college_pitchers.csv', index=False)

print("Sample dataset created")
print("Total pitches:", len(sample_df))
print("Pitchers:", sample_df['player_name'].nunique())
