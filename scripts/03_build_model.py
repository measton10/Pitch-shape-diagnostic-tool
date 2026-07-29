"""
Trains the final shape -> expected whiff rate model.

This is a Random Forest classifier (chosen after logistic regression
plateaued at AUC ~0.57 regardless of feature scaling - see
project_log.md, Steps 5-9 for the full debugging process).

Inputs: pitch shape metrics + pitch type (one-hot encoded)
Output: predicted probability that a given pitch results in a whiff
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('league_sample.csv')
df['is_whiff'] = df['description'] == 'swinging_strike'

SHAPE_COLS = ['release_speed', 'release_spin_rate', 'spin_axis',
              'pfx_x', 'pfx_z', 'release_extension']

df = df.dropna(subset=SHAPE_COLS + ['pitch_type'])

pitch_type_dummies = pd.get_dummies(df['pitch_type'], prefix='type')
X = pd.concat([df[SHAPE_COLS], pitch_type_dummies], axis=1)
y = df['is_whiff']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, predictions)

print("Model trained successfully")
print("AUC score (0.5 = random guessing, 1.0 = perfect):", round(score, 3))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 most important features:")
print(importances.head(10))
