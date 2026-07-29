"""
Validation 1: Paul Skenes (Pirates), 2024 season.

Real scouting consensus holds Skenes' changeup as an elite, standout
pitch. This script checks whether the model independently identifies
the same thing from the numbers alone.
"""

from pybaseball import statcast_pitcher
from model_utils import train_model, score_pitcher

model, pitch_type_columns = train_model()

skenes = statcast_pitcher('2024-04-01', '2024-09-30', player_id=694973)
summary = score_pitcher(skenes, model, pitch_type_columns)

print("PAUL SKENES - Predicted vs Actual Whiff Rate + Location")
print("=========================================================")
for pitch_type, row in summary.iterrows():
    gap = round(row['actual_whiff_rate'] - row['predicted_whiff_rate'], 3)
    print(f"\n{pitch_type} - {int(row['pitch_count'])} pitches")
    print(f"  actual whiff rate:    {row['actual_whiff_rate']}")
    print(f"  predicted whiff rate: {row['predicted_whiff_rate']}")
    print(f"  gap:                  {gap}")
    print(f"  chase zone pct:       {row['pct_chase_zone']}")
    print(f"  heart zone pct:       {row['pct_heart_zone']}")
