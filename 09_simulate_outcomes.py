"""
Applies the trained model to the simulated sample pitchers, then
generates simulated actual outcomes using per-pitcher "skill" and
"command" modifiers - standing in for real-world factors (deception,
command) the model can't observe directly. This produces realistic,
varied gap patterns for the demo.
"""

import pandas as pd
import numpy as np
from model_utils import train_model

np.random.seed(42)

model, pitch_type_columns = train_model()

sample = pd.read_csv('sample_college_pitchers.csv')

from model_utils import SHAPE_COLS
sample_types = pd.get_dummies(sample['pitch_type'], prefix='type')
sample_types = sample_types.reindex(columns=pitch_type_columns, fill_value=0)
X_sample = pd.concat([sample[SHAPE_COLS], sample_types], axis=1)

sample['predicted_whiff_prob'] = model.predict_proba(X_sample)[:, 1]

# Skill modifiers simulate deception/command effects beyond raw shape
SKILL_MODIFIERS = {
    'Sample Pitcher 1': 1.6,
    'Sample Pitcher 2': 0.6,
    'Sample Pitcher 3': 1.0,
    'Sample Pitcher 4': 1.2,
    'Sample Pitcher 5': 0.8,
}
sample['skill_modifier'] = sample['player_name'].map(SKILL_MODIFIERS)
sample['adjusted_prob'] = (sample['predicted_whiff_prob'] * sample['skill_modifier']).clip(0, 1)
sample['is_whiff'] = np.random.binomial(1, sample['adjusted_prob'])

# Command modifiers simulate location quality independent of shape
COMMAND_MODIFIERS = {
    'Sample Pitcher 1': 0.85,
    'Sample Pitcher 2': 0.35,
    'Sample Pitcher 3': 0.55,
    'Sample Pitcher 4': 0.60,
    'Sample Pitcher 5': 0.45,
}
sample['command_score'] = sample['player_name'].map(COMMAND_MODIFIERS)
sample['pct_chase_zone'] = (sample['command_score'] + np.random.normal(0, 0.08, len(sample))).clip(0.1, 0.95)

sample.to_csv('sample_college_pitchers_with_outcomes.csv', index=False)

sample['gap'] = sample['is_whiff'] - sample['predicted_whiff_prob']

summary = sample.groupby(['player_name', 'pitch_type']).agg(
    predicted_whiff_rate=('predicted_whiff_prob', 'mean'),
    actual_whiff_rate=('is_whiff', 'mean')
).round(3)
summary['gap'] = (summary['actual_whiff_rate'] - summary['predicted_whiff_rate']).round(3)

print(summary)

pitcher_summary = sample.groupby('player_name').agg(
    avg_gap=('gap', 'mean'),
    avg_chase_zone_pct=('pct_chase_zone', 'mean')
).round(3)

print("\nPitcher-Level Summary: Performance Gap vs. Command")
print(pitcher_summary)
