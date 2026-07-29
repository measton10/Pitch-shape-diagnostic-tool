"""
Standalone location profile check - shows whiff rate alongside chase-zone
and heart-zone percentages for a single pitcher (default: Skenes), without
needing the full model. Useful for quick location-only checks.
"""

import pandas as pd

df = pd.read_csv('skenes_trimmed.csv')
df['is_whiff'] = df['description'] == 'swinging_strike'

# Statcast zones: 1-9 are in the strike zone, 11-14 are "chase" zones
df['in_chase_zone'] = df['zone'].isin([11, 12, 13, 14])
df['in_heart_zone'] = df['zone'].isin([5])  # dead center - most hittable

location_summary = df.groupby('pitch_type').agg(
    whiff_rate=('is_whiff', 'mean'),
    pct_chase_zone=('in_chase_zone', 'mean'),
    pct_heart_zone=('in_heart_zone', 'mean'),
    pitch_count=('is_whiff', 'count')
).round(3).sort_values('pitch_count', ascending=False)

print("Location Profile by Pitch Type:")
print(location_summary)
