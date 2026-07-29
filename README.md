# Pitch Shape vs. Performance Diagnostic Tool

A model that separates *what a pitch's shape predicts* from *what it actually produces* — and when there's a gap, diagnoses whether it's a command problem, a shape problem, or something else entirely. Built as a baseball analytics + AI portfolio project, targeting a student analyst/manager role with Miami University baseball.

<img width="1710" height="1107" alt="Screenshot 2026-07-28 at 10 44 49 PM" src="https://github.com/user-attachments/assets/161e8244-6591-470e-8852-ffb503862055" />


## The idea

Pitching coaches already do this informally: look at a pitch's shape (spin, break, velocity), decide if it "should" be effective, then compare that to what actually happens in games. This project systematizes that process using real MLB Statcast data, and adds a second AI-driven layer — using Claude to turn the numbers into an actual coaching diagnosis and recommendation, not just a stat line.

## How it works

1. **Train a baseline model** on real MLB pitch data — given a pitch's shape (velocity, spin rate, spin axis, horizontal/vertical break, extension, pitch type), predict the expected whiff rate.
2. **Compare predicted vs. actual** for a given pitcher — the gap tells you whether a pitch is over- or under-performing its shape.
3. **Add a location layer** — using strike zone data, check whether a pitcher is locating the pitch well (chase zone) or poorly (heart of the plate). This separates *shape* problems from *command* problems.
4. **Feed the results to Claude** to generate a real diagnosis and recommendation — e.g., "this pitcher's overperformance is command-driven, protect his mechanics" vs. "this underperformance isn't explained by location — something else is going on."

## Model performance

- Trained on **116,355 real MLB pitches** across 531 pitchers (June 2024)
- Random Forest classifier, **AUC 0.61** (vs. 0.51 baseline/random guessing)
- Debugged through several iterations — see `project_log.md` for the full process, including dead ends and why they happened

## Validation — tested against 3 real MLB pitchers with known reputations

| Pitcher | Pattern Found | Explanation |
|---|---|---|
| **Paul Skenes** | Changeup massively overperforms shape (+14.3%) | Elite command — 78.9% chase zone, 0% heart zone. Matches real scouting consensus on his changeup being an elite weapon. |
| **Framber Valdez** | Sinker underperforms, curveball/changeup overperform | Sinker is a contact-management pitch by design, not a whiff pitch — matches his real pitching identity. Curveball (his signature pitch) has the best location profile. |
| **Kyle Hendricks** | Every pitch underperforms shape | Matches his real, well-documented rough 2024 season (demoted to bullpen, 10.57 ERA). Notably, his location wasn't bad — a genuine finding that this model's two layers don't explain every case, likely due to declining stuff or reduced deception with age. |

The model correctly read three structurally different pitcher archetypes for three different, verifiable reasons — including honestly identifying a case its current features can't fully explain.

## Repo structure

```
├── README.md                          — this file
├── project_log.md                     — full build log: every step, decision, and dead end
├── pitch_gap_demo.html                — interactive demo (open in any browser)
├── scripts/
│   ├── 01_pull_pitcher_data.py        — pull single-pitcher Statcast data
│   ├── 02_pull_league_data.py         — pull league-wide reference data
│   ├── 03_build_model.py              — train the shape → whiff rate model
│   ├── 04_validate_skenes.py          — validation: Paul Skenes
│   ├── 05_validate_valdez.py          — validation: Framber Valdez
│   ├── 06_validate_hendricks.py       — validation: Kyle Hendricks
│   ├── 07_location_analysis.py        — chase zone / heart zone location layer
│   ├── 08_sample_pitchers.py          — simulated college-level sample data
│   └── 09_simulate_outcomes.py        — simulated outcomes + gap calculation for sample data
```

## Tech stack

- **Data:** MLB Statcast via [`pybaseball`](https://github.com/jldbc/pybaseball)
- **Model:** scikit-learn (`RandomForestClassifier`)
- **Reasoning layer:** Claude (Anthropic)
- **Demo:** vanilla HTML/CSS/JS

## Honest limitations

- The model explains roughly moderate variance in whiff outcomes (AUC 0.61) — shape and location are real, meaningful factors, but not the whole story (see Hendricks case above).
- Sample college pitcher data is simulated, not real TrackMan data, since real program data wasn't available during this build. The pipeline is designed to accept real TrackMan-format data directly.
- The Claude reasoning step is currently a manual process (data generated in Python, reasoned over in conversation) rather than a fully automated API call.

## About

Built by Matt Weaston, rising senior at Miami University (Business Analytics major, Sport Management minor), as a portfolio project combining sports analytics and AI.
