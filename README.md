# Pitch Shape vs. Performance Diagnostic Tool

**[View the live interactive site →](https://measton10.github.io/Pitch-shape-diagnostic-tool/)**

A diagnostic tool that separates what a pitch's physical shape predicts from what it actually produces — and when there's a gap, figures out whether it's a command problem, a shape problem, a matchup problem, or something else entirely. Built as a baseball analytics + AI portfolio project, targeting a student analyst/manager role with Miami University baseball.

## The idea

A pitching coach looks at a pitch's spin, break, and velocity and asks: "should this pitch be good?" Then they watch what actually happens in games and ask: "is it living up to that?" This project systematizes that process using real MLB Statcast data — four independent outcome models, a location/command layer, real platoon splits by batter handedness, and a composite grading system — then uses Claude to turn the numbers into actual coaching diagnoses and recommendations, not just a stat line.

## How it works

1. **Four outcome models** — given a pitch's shape (velocity, spin rate, spin axis, break, extension, pitch type), predict:
   - **Whiff rate** — chance of a swing and miss
   - **Hard-hit rate** — chance of 95+ mph exit velocity
   - **Ground ball rate** — chance of a low launch angle (weak/managed contact)
   - **Barrel rate** — chance of meeting MLB's official "barrel" standard (exit velocity + launch angle combined), the strictest damaging-contact measure
2. **Compare predicted vs. actual** for a given pitcher — the gap on each model tells you whether a pitch is over- or under-performing its shape, and in what specific way.
3. **Location layer** — using strike-zone data, check whether a pitcher is locating well (chase zone) or poorly (heart of the plate), separating shape problems from command problems.
4. **Platoon splits** — the same gap analysis run separately against left-handed and right-handed batters, since a pitch can be a real weapon against one side and nearly unusable against the other.
5. **Composite grade** — all four model gaps converted to z-scores against the league-wide distribution and averaged into one number, scaled so 100 = league average (like real tools such as Stuff+/PitchingBot).
6. **Claude reasoning layer** — takes the numbers and produces a real diagnosis and recommendation, including situational, coach-usable "game plans" per pitcher.

## Model performance

- Trained on **116,355+ real MLB pitches** (whiff model) and **20,000-38,000+ balls in play** (contact models), all from June 2024
- Four Random Forest classifiers:

| Model | AUC (0.5 = random, 1.0 = perfect) |
|---|---|
| Whiff | 0.610 |
| Hard-Hit | 0.546 |
| Ground Ball | 0.691 |
| Barrel | 0.563 |

- Debugged through several iterations — see `project_log.md` for the full process, including dead ends, a data-scaling fix, and switching from logistic regression to Random Forest

## Validation — tested against 3 real MLB pitchers with contrasting profiles

| Pitcher | Archetype | Key Finding |
|---|---|---|
| **Paul Skenes** | Swing-and-miss ace | Changeup grades 121 (best in arsenal), driven by elite command (78.9% chase zone) — **not** tunneling off the fastball, an early assumption the data disproved. Slider and Valdez's slider both show a real barrel-rate risk pattern. |
| **Framber Valdez** | Contact-management sinkerballer | Sinker allows harder contact than predicted but induces far more ground balls — the metric that actually matters for a pitch designed to limit damage, not miss bats. Changeup nearly doubles its whiff rate against lefties, confirming a classic scouting principle with real data. |
| **Kyle Hendricks** | A real, documented decline | Every pitch underperforms, matching his real rough 2024 (10.57 ERA, demoted to bullpen). The barrel model pinpoints the cause: his fastball is barreled at more than double its predicted rate (+12.6 points, the largest gap in this project) — concentrated specifically against left-handed batters. |

The model correctly read three structurally different pitcher archetypes for three different, verifiable reasons — including self-correcting an earlier wrong assumption and honestly flagging a case (Hendricks' broader decline) its current features can't fully explain.

## Repo structure

```
├── README.md                          — this file
├── project_log.md                     — full build log: every step, decision, and dead end
├── docs/                               — the live 4-page interactive site (GitHub Pages)
│   ├── index.html                      — Home
│   ├── methodology.html                — How It Works
│   ├── validation.html                 — Validation results (3 real pitchers)
│   └── recommendations.html            — Interactive tool + coach-usable game plans
├── scripts/
│   ├── 01_pull_pitcher_data.py         — pull single-pitcher Statcast data
│   ├── 02_pull_league_data.py          — pull league-wide reference data
│   ├── 03_build_model.py               — train the whiff rate model
│   ├── 04_validate_skenes.py           — validation: Paul Skenes
│   ├── 05_validate_valdez.py           — validation: Framber Valdez
│   ├── 06_validate_hendricks.py        — validation: Kyle Hendricks
│   ├── 07_location_analysis.py         — chase zone / heart zone command layer
│   ├── 08_sample_pitchers.py           — simulated sample pitcher data (early prototype)
│   ├── 09_simulate_outcomes.py         — simulated outcomes + gap calculation
│   ├── 10_build_barrel_model.py        — barrel rate model (MLB's official standard)
│   ├── 11_platoon_splits.py            — lefty vs. righty whiff gap analysis
│   ├── 12_final_composite_grade.py     — full 4-model composite grading system
│   └── model_utils.py                  — shared training/scoring helper functions
```

## Tech stack

- **Data:** MLB Statcast via [`pybaseball`](https://github.com/jldbc/pybaseball)
- **Models:** scikit-learn (`RandomForestClassifier`)
- **Reasoning layer:** Claude (Anthropic)
- **Site:** vanilla HTML/CSS/JS, hosted on GitHub Pages — no framework, no build step

## Honest limitations

- The hard-hit model (AUC 0.546) is the weakest of the four — contact quality carries more inherent randomness (defense, luck) than swing-and-miss or batted-ball angle.
- Tunneling analysis uses release-point proximity as a proxy for true pitch-flight similarity, since frame-by-frame tracking data wasn't available.
- League-wide baselines are built on one month of 2024 data (June); a full season would produce more stable distributions.
- Small-sample results (e.g., a pitcher's rarely-thrown pitch) are automatically flagged with a confidence label rather than reported at face value.
- Sample college-pitcher data (scripts 08-09) is simulated, not real TrackMan data, since a real program feed wasn't available during this build — the pipeline is designed to accept that format directly if it becomes available.
- Hendricks' broader 2024 decline isn't fully explained by shape, location, or platoon splits — likely involves pitch recognition or reduced deception, factors outside this tool's current feature set. The tool flags this honestly rather than forcing an explanation.
- The Claude reasoning layer is currently a manual process (model output reasoned over in conversation, then written into the site) rather than a live API call — a deliberate scope decision given the project timeline, documented in `project_log.md`.

## About

Built by Matt Easton, rising senior at Miami University (Business Analytics major, Sport Management minor), as a portfolio project combining sports analytics and AI.
