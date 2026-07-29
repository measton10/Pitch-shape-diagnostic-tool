"""
Validation 2: Framber Valdez (Astros), 2024 season.

Valdez is a contact-management/groundball pitcher, not a swing-and-miss
pitcher - a structurally different archetype from Skenes. This checks
whether the model correctly reads a very different kind of pitcher.
"""

from pybaseball import statcast_pitcher, playerid_lookup
from model_utils import train_model, score_pitcher

model, pitch_type_columns = train_model()

lookup = playerid_lookup('valdez', 'framber')
player_id = int(lookup['key_mlbam'].iloc[0])

valdez = statcast_pitcher('2024-04-01', '2024-09-30', player_id=player_id)
summary = score_pitcher(valdez, model, pitch_type_columns)

print("FRAMBER VALDEZ - Predicted vs Actual Whiff Rate + Location")
print("=============================================================")
for pitch_type, row in summary.iterrows():
    gap = round(row['actual_whiff_rate'] - row['predicted_whiff_rate'], 3)
    print(f"\n{pitch_type} - {int(row['pitch_count'])} pitches")
    print(f"  actual whiff rate:    {row['actual_whiff_rate']}")
    print(f"  predicted whiff rate: {row['predicted_whiff_rate']}")
    print(f"  gap:                  {gap}")
    print(f"  chase zone pct:       {row['pct_chase_zone']}")
    print(f"  heart zone pct:       {row['pct_heart_zone']}")
