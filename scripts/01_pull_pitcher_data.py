"""
Pulls a single pitcher's real Statcast data for one season.
Used as the first exploratory step to understand the data structure,
and later reused as the basis for the validation scripts.
"""

from pybaseball import statcast_pitcher
import pandas as pd

# Paul Skenes (Pirates) - MLBAM ID 694973
data = statcast_pitcher('2024-04-01', '2024-09-30', player_id=694973)

# Keep only the columns needed for shape + outcome analysis
key_cols = ['pitch_type', 'release_speed', 'release_spin_rate', 'spin_axis',
            'pfx_x', 'pfx_z', 'release_extension', 'zone', 'description']
df = data[key_cols].copy()

df.to_csv('skenes_trimmed.csv', index=False)

# Quick check: whiff rate by pitch type
df['is_whiff'] = df['description'] == 'swinging_strike'
whiff_rate = df.groupby('pitch_type')['is_whiff'].mean().sort_values(ascending=False)

print("Whiff rate by pitch type:")
print(whiff_rate)
