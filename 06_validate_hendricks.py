"""
Validation 3: Kyle Hendricks (Cubs), 2024 season.

Hendricks had a well-documented, genuinely poor 2024 (demoted to the
bullpen after a 10.57 ERA start). This checks whether the model
correctly flags a real decline - and where its explanation runs out.
"""

from pybaseball import statcast_pitcher, playerid_lookup
from model_utils import train_model, score_pitcher

model, pitch_type_columns = train_model()

lookup = playerid_lookup('hendricks', 'kyle')
player_id = int(lookup['key_mlbam'].iloc[0])

hendricks = statcast_pitcher('2024-04-01', '2024-09-30', player_id=player_id)
summary = score_pitcher(hendricks, model, pitch_type_columns)

print("KYLE HENDRICKS - Predicted vs Actual Whiff Rate + Location")
print("=============================================================")
for pitch_type, row in summary.iterrows():
    gap = round(row['actual_whiff_rate'] - row['predicted_whiff_rate'], 3)
    print(f"\n{pitch_type} - {int(row['pitch_count'])} pitches")
    print(f"  actual whiff rate:    {row['actual_whiff_rate']}")
    print(f"  predicted whiff rate: {row['predicted_whiff_rate']}")
    print(f"  gap:                  {gap}")
    print(f"  chase zone pct:       {row['pct_chase_zone']}")
    print(f"  heart zone pct:       {row['pct_heart_zone']}")
