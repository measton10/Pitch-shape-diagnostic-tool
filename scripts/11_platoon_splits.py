"""
Platoon splits: applies the validated whiff model separately to left-handed
and right-handed batters, to check whether a pitch's shape-vs-actual gap
changes depending on who's in the batter's box. None of the other models
in this project account for batter handedness.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pybaseball import statcast_pitcher, playerid_lookup

SHAPE_COLS = ['release_speed', 'release_spin_rate', 'spin_axis',
              'pfx_x', 'pfx_z', 'release_extension']

whiff_data = pd.read_csv('league_sample.csv')
whiff_data['is_whiff'] = whiff_data['description'] == 'swinging_strike'
whiff_data = whiff_data.dropna(subset=SHAPE_COLS + ['pitch_type'])

pitch_type_dummies = pd.get_dummies(whiff_data['pitch_type'], prefix='type')
X_train = pd.concat([whiff_data[SHAPE_COLS], pitch_type_dummies], axis=1)
y_train = whiff_data['is_whiff']

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)


def platoon_check(name, player_id):
    pdata = statcast_pitcher('2024-04-01', '2024-09-30', player_id=player_id)
    pdata = pdata.dropna(subset=SHAPE_COLS + ['pitch_type', 'stand'])
    pdata['is_whiff'] = pdata['description'] == 'swinging_strike'

    types = pd.get_dummies(pdata['pitch_type'], prefix='type')
    types = types.reindex(columns=pitch_type_dummies.columns, fill_value=0)
    X = pd.concat([pdata[SHAPE_COLS], types], axis=1)
    pdata['predicted'] = model.predict_proba(X)[:, 1]

    print(f"\n{'='*50}\n{name.upper()} - PLATOON SPLITS (Whiff Rate)\n{'='*50}")

    for pt in pdata['pitch_type'].unique():
        sub = pdata[pdata['pitch_type'] == pt]
        if len(sub) < 30:
            continue

        vs_L = sub[sub['stand'] == 'L']
        vs_R = sub[sub['stand'] == 'R']
        if len(vs_L) < 15 or len(vs_R) < 15:
            continue

        gap_L = vs_L['is_whiff'].mean() - vs_L['predicted'].mean()
        gap_R = vs_R['is_whiff'].mean() - vs_R['predicted'].mean()

        print(f"\n{pt} - vs L: {len(vs_L)} pitches | vs R: {len(vs_R)} pitches")
        print(f"  whiff rate vs LHH: {round(vs_L['is_whiff'].mean(), 3)}  gap: {round(gap_L, 3)}")
        print(f"  whiff rate vs RHH: {round(vs_R['is_whiff'].mean(), 3)}  gap: {round(gap_R, 3)}")
        print(f"  platoon difference (L minus R): {round(gap_L - gap_R, 3)}")


platoon_check("Paul Skenes", 694973)

lookup = playerid_lookup('valdez', 'framber')
platoon_check("Framber Valdez", int(lookup['key_mlbam'].iloc[0]))

lookup = playerid_lookup('hendricks', 'kyle')
platoon_check("Kyle Hendricks", int(lookup['key_mlbam'].iloc[0]))
