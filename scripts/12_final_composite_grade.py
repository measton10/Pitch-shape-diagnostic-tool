"""
Final composite grading system: combines all four outcome models (whiff,
hard-hit, ground ball, barrel) into one grade per pitch, scaled so
100 = league average, similar to real tools like Stuff+/PitchingBot.
Each pitcher's gap on each model is converted to a z-score relative to
the league-wide distribution for that pitch type, then the four z-scores
are averaged into the final grade.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pybaseball import statcast_pitcher, playerid_lookup

SHAPE_COLS = ['release_speed', 'release_spin_rate', 'spin_axis',
              'pfx_x', 'pfx_z', 'release_extension']


def train(target_col, data):
    d = data.dropna(subset=SHAPE_COLS + ['pitch_type'])
    dummies = pd.get_dummies(d['pitch_type'], prefix='type')
    X = pd.concat([d[SHAPE_COLS], dummies], axis=1)
    y = d[target_col]
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X, y)
    return model, dummies.columns


# Train all four models
whiff_data = pd.read_csv('league_sample.csv')
whiff_data['is_whiff'] = whiff_data['description'] == 'swinging_strike'
whiff_model, whiff_cols = train('is_whiff', whiff_data)

contact_data = pd.read_csv('league_sample_with_contact.csv')
hh_data = contact_data.dropna(subset=['launch_speed']).copy()
hh_data['is_hard_hit'] = hh_data['launch_speed'] >= 95
hh_model, hh_cols = train('is_hard_hit', hh_data)

gb_data = contact_data.dropna(subset=['launch_angle']).copy()
gb_data['is_groundball'] = gb_data['launch_angle'] < 10
gb_model, gb_cols = train('is_groundball', gb_data)

barrel_data = pd.read_csv('league_sample_with_barrel.csv')
barrel_data = barrel_data.dropna(subset=['launch_speed_angle'])
barrel_data['is_barrel'] = barrel_data['launch_speed_angle'] == 6
barrel_model, barrel_cols = train('is_barrel', barrel_data)

# League-wide gap distributions (mean/std by pitch type) used to convert
# any pitcher's raw gap into a z-score. See scripts for how each
# *_gaps.csv file is built - the same predicted-vs-actual gap calculation
# run across every pitcher in the league sample, then grouped by pitch type.
whiff_dist = pd.read_csv('league_whiff_gaps.csv').groupby('pitch_type')['gap'].agg(['mean', 'std'])
hh_dist = pd.read_csv('league_hardhit_gaps.csv').groupby('pitch_type')['gap'].agg(['mean', 'std'])
gb_dist = pd.read_csv('league_gb_gaps.csv').groupby('pitch_type')['gap'].agg(['mean', 'std'])
barrel_dist = pd.read_csv('league_barrel_gaps.csv').groupby('pitch_type')['gap'].agg(['mean', 'std'])


def predict(model, cols, d):
    dummies = pd.get_dummies(d['pitch_type'], prefix='type').reindex(columns=cols, fill_value=0)
    X = pd.concat([d[SHAPE_COLS], dummies], axis=1)
    return model.predict_proba(X)[:, 1]


def grade_pitcher(name, player_id):
    pdata = statcast_pitcher('2024-04-01', '2024-09-30', player_id=player_id)
    pdata = pdata.dropna(subset=SHAPE_COLS + ['pitch_type'])

    pdata['is_whiff'] = pdata['description'] == 'swinging_strike'
    pdata['whiff_pred'] = predict(whiff_model, whiff_cols, pdata)

    hh_p = pdata.dropna(subset=['launch_speed']).copy()
    hh_p['is_hard_hit'] = hh_p['launch_speed'] >= 95
    hh_p['hh_pred'] = predict(hh_model, hh_cols, hh_p)

    gb_p = pdata.dropna(subset=['launch_angle']).copy()
    gb_p['is_groundball'] = gb_p['launch_angle'] < 10
    gb_p['gb_pred'] = predict(gb_model, gb_cols, gb_p)

    barrel_p = pdata.dropna(subset=['launch_speed_angle']).copy()
    barrel_p['is_barrel'] = barrel_p['launch_speed_angle'] == 6
    barrel_p['barrel_pred'] = predict(barrel_model, barrel_cols, barrel_p)

    print(f"\n{'='*45}\n{name.upper()} - 4-MODEL GRADES\n{'='*45}")

    for pt in pdata['pitch_type'].unique():
        w = pdata[pdata['pitch_type'] == pt]
        if len(w) < 20:
            continue
        whiff_gap = w['is_whiff'].mean() - w['whiff_pred'].mean()

        h = hh_p[hh_p['pitch_type'] == pt]
        hh_gap = (h['is_hard_hit'].mean() - h['hh_pred'].mean()) if len(h) >= 10 else None

        g = gb_p[gb_p['pitch_type'] == pt]
        gb_gap = (g['is_groundball'].mean() - g['gb_pred'].mean()) if len(g) >= 10 else None

        b = barrel_p[barrel_p['pitch_type'] == pt]
        barrel_gap = (b['is_barrel'].mean() - b['barrel_pred'].mean()) if len(b) >= 10 else None

        # z-scores: whiff and gb are "positive is good"; hard-hit and barrel
        # are "negative is good", so those two are flipped before averaging
        z_whiff = (whiff_gap - whiff_dist.loc[pt, 'mean']) / whiff_dist.loc[pt, 'std'] if pt in whiff_dist.index else 0
        z_hh = -((hh_gap - hh_dist.loc[pt, 'mean']) / hh_dist.loc[pt, 'std']) if hh_gap is not None and pt in hh_dist.index else 0
        z_gb = (gb_gap - gb_dist.loc[pt, 'mean']) / gb_dist.loc[pt, 'std'] if gb_gap is not None and pt in gb_dist.index else 0
        z_barrel = -((barrel_gap - barrel_dist.loc[pt, 'mean']) / barrel_dist.loc[pt, 'std']) if barrel_gap is not None and pt in barrel_dist.index else 0

        composite_z = (z_whiff + z_hh + z_gb + z_barrel) / 4
        grade = round(100 + (composite_z * 15))

        print(f"\n{pt} ({len(w)} pitches) - GRADE: {grade}")
        print(f"  whiff gap: {round(whiff_gap, 3)}")
        print(f"  hard-hit gap: {round(hh_gap, 3) if hh_gap is not None else 'n/a'}")
        print(f"  gb gap: {round(gb_gap, 3) if gb_gap is not None else 'n/a'}")
        print(f"  barrel gap: {round(barrel_gap, 4) if barrel_gap is not None else 'n/a'}")


grade_pitcher("Paul Skenes", 694973)

lookup = playerid_lookup('valdez', 'framber')
grade_pitcher("Framber Valdez", int(lookup['key_mlbam'].iloc[0]))

lookup = playerid_lookup('hendricks', 'kyle')
grade_pitcher("Kyle Hendricks", int(lookup['key_mlbam'].iloc[0]))
