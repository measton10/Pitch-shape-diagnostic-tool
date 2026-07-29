"""
Shared helper: trains the shape -> whiff rate model on league data.
Imported by each validation script so the training logic lives in one
place instead of being copy-pasted across files.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SHAPE_COLS = ['release_speed', 'release_spin_rate', 'spin_axis',
              'pfx_x', 'pfx_z', 'release_extension']


def train_model(league_csv='league_sample.csv'):
    league = pd.read_csv(league_csv)
    league['is_whiff'] = league['description'] == 'swinging_strike'
    league = league.dropna(subset=SHAPE_COLS + ['pitch_type'])

    pitch_type_dummies = pd.get_dummies(league['pitch_type'], prefix='type')
    X_train = pd.concat([league[SHAPE_COLS], pitch_type_dummies], axis=1)
    y_train = league['is_whiff']

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    return model, pitch_type_dummies.columns


def score_pitcher(pitcher_df, model, pitch_type_columns):
    """Given a pitcher's raw Statcast data, adds predicted whiff probability
    and location zone flags, and returns a pitch-type-level summary."""
    pitcher_df = pitcher_df.copy()
    pitcher_df['is_whiff'] = pitcher_df['description'] == 'swinging_strike'
    pitcher_df = pitcher_df.dropna(subset=SHAPE_COLS + ['pitch_type'])

    pitcher_df['in_chase_zone'] = pitcher_df['zone'].isin([11, 12, 13, 14])
    pitcher_df['in_heart_zone'] = pitcher_df['zone'].isin([5])

    type_dummies = pd.get_dummies(pitcher_df['pitch_type'], prefix='type')
    type_dummies = type_dummies.reindex(columns=pitch_type_columns, fill_value=0)
    X = pd.concat([pitcher_df[SHAPE_COLS], type_dummies], axis=1)

    pitcher_df['predicted_whiff_prob'] = model.predict_proba(X)[:, 1]

    summary = pitcher_df.groupby('pitch_type').agg(
        actual_whiff_rate=('is_whiff', 'mean'),
        predicted_whiff_rate=('predicted_whiff_prob', 'mean'),
        pct_chase_zone=('in_chase_zone', 'mean'),
        pct_heart_zone=('in_heart_zone', 'mean'),
        pitch_count=('is_whiff', 'count')
    ).round(3).sort_values('pitch_count', ascending=False)

    return summary
