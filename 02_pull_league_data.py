"""
Pulls league-wide MLB Statcast data to serve as the training set for the
shape-to-whiff-rate model. One month of data (June 2024) was used as the
final version, after testing showed one week of data was too small a
sample for a stable model (see project_log.md, Step 9).
"""

from pybaseball import statcast
import pandas as pd

data = statcast(start_dt='2024-06-01', end_dt='2024-06-30')

key_cols = ['pitcher', 'player_name', 'pitch_type', 'release_speed',
            'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
            'release_extension', 'zone', 'description']
df = data[key_cols].copy()

df.to_csv('league_sample.csv', index=False)

df['is_whiff'] = df['description'] == 'swinging_strike'

print("Total pitches pulled:", len(df))
print("Unique pitchers in sample:", df['player_name'].nunique())
