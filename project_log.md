# Project Log — Pitch Shape vs. Performance

A condensed build log covering the major phases of this project, what was done at each stage, and what was found. For the full step-by-step history (including debugging dead ends), see the git commit history.

---

## Phase 1 — Getting the Data

**What we did:** Set up Python locally (`pybaseball`, `pandas`, `scikit-learn`), then pulled real MLB Statcast data:
- A single pitcher's full-season data (Paul Skenes, 2024) to understand the data's structure — 2,127 pitches, 119 columns.
- A league-wide reference sample (June 2024) to train models against — eventually expanded from one week (26,235 pitches) to a full month (116,355 pitches, 531 pitchers) once early testing showed a bigger sample stabilized results.

**Key columns used:** velocity, spin rate, spin axis, horizontal/vertical break, release extension, pitch type (the "shape" inputs), plus pitch outcome, exit velocity, launch angle, and Statcast's official barrel classification (the "results" to predict).

---

## Phase 2 — Building the Whiff Rate Model

**What we did:** Trained a model to predict the chance a pitch gets swung on and missed, based on shape alone. Compared each pitcher's actual whiff rate against the model's prediction — the difference is the "gap" used throughout this project.

**What we found:** Early versions barely beat random guessing (AUC ~0.51). Two fixes mattered: (1) telling the model which pitch type it was looking at, since "good" spin rate means different things for a fastball vs. a curveball, and (2) using a Random Forest instead of a straight-line model, since the real relationship between shape and results isn't linear. Final model: **AUC 0.610**, trained on 116,355 pitches.

---

## Phase 3 — Real Pitcher Validation

**What we did:** Applied the model to three real 2024 MLB pitchers, chosen specifically for contrasting reputations, to check whether the model's findings matched real scouting truth.

**What we found:**
- **Paul Skenes** (swing-and-miss ace) — his changeup overperformed its shape by a wide margin (+14.3 whiff points), landing in the chase zone 78.9% of the time and the hittable heart zone 0% of the time. Matches his real reputation as having an elite changeup.
- **Framber Valdez** (contact-management sinkerballer) — his sinker underperformed on whiff rate, matching his real identity as a groundball pitcher, not a strikeout pitcher.
- **Kyle Hendricks** (real decline) — every pitch underperformed, matching his actual rough 2024 season (10.57 ERA, demoted to the bullpen).

This confirmed the model reads real, structurally different pitchers correctly — not just fitting one lucky case.

---

## Phase 4 — Adding a Location/Command Layer

**What we did:** Used Statcast's strike-zone data to check whether a pitcher is locating well (the "chase zone," just off the plate) or poorly (the "heart zone," dead center) — separating command problems from shape problems.

**What we found:** Skenes' changeup succeeds specifically because of elite command (78.9% chase zone), not just good shape. This let the diagnosis go from one-dimensional ("is this pitch good?") to two-dimensional ("is it good because of shape, or because of command?").

---

## Phase 5 — Expanding Beyond Whiff Rate

**What we did:** Whiff rate alone can't fairly judge a pitch like Valdez's sinker, which isn't designed to miss bats. Built two more models: **hard-hit rate** (95+ mph contact) and **ground ball rate** (batted-ball angle).

**What we found:** Valdez's sinker allows harder contact than predicted, but also induces far more ground balls than predicted — the metric that actually matters for a pitch built to limit damage. A single metric would have misjudged this pitch entirely; three together told the real story.

---

## Phase 6 — The Composite Grade

**What we did:** Combined all outcome models into one number per pitch, scaled so **100 = league average** (the same approach real tools like Stuff+ use) — each model's gap converted to a z-score against the league-wide distribution, then averaged.

**What we found:** The composite grade independently reproduced findings already discovered by hand (Skenes' changeup grading highest, Valdez's curveball as his signature pitch), and correctly kept Hendricks' grades close to league-average rather than artificially tanking them just because his season was bad — a sign the system stayed honest rather than overfitting to a known outcome.

---

## Phase 7 — Key Finding: The Tunneling Self-Correction

**What we did:** An early diagnosis assumed Skenes' changeup succeeded because it "tunneled" off his fastball (near-identical release point, making it hard to pick up early). Tested this directly by measuring release-point distance between every pair of his pitches.

**What we found:** The changeup–fastball pair was actually one of the *worst*-matched pairs in his arsenal. The real explanation was command, not deception — his true best-tunneling pair is fastball–splitter. Correcting this assumption once better evidence arrived was a deliberate choice to prioritize accuracy over the original narrative.

---

## Phase 8 — Key Finding: The 2-Strike Usage Gap

**What we did:** Compared each pitch's overall usage rate to its usage specifically in 2-strike counts, alongside how much each pitch's whiff rate improves in that situation ("putaway edge").

**What we found:** Skenes' splitter improves the most of any pitch in 2-strike counts (+5.8%), yet is thrown ~13 percentage points *less* often in exactly those counts — while his fastball, which doesn't improve at all, gets thrown ~10 points more. A precise, actionable usage recommendation.

---

## Phase 9 — Adding Barrel Rate (4th Model)

**What we did:** Added a model for MLB's official "barrel" classification — a stricter combination of exit velocity and launch angle than hard-hit rate alone.

**What we found:** This model changed real conclusions, not just added a number. Skenes' sweeper (previously his weakest-graded pitch) actually suppresses barrels well, pulling its grade up. Most notably, Hendricks' fastball is barreled at **more than double** its predicted rate (+12.6 points) — the largest gap found anywhere in this project, and a precise, concrete explanation for his decline that earlier models could only describe in general terms.

---

## Phase 10 — Platoon Splits

**What we did:** Split each validated pitcher's real data by batter handedness and re-ran the whiff-gap analysis separately for lefties and righties.

**What we found:**
- Skenes' sweeper has **0% whiff rate in 50 pitches against lefties** this season, but performs normally against righties.
- Skenes' slider is nearly 3x more effective against lefties than righties.
- Valdez's changeup nearly doubles its whiff rate against lefties — confirming a classic scouting principle (a same-handed pitcher's changeup plays up against opposite-handed batters) with real data.
- Hendricks' fastball problem is concentrated specifically against lefties — it actually beats its shape prediction against righties, adding precision to the barrel-rate finding above.

---

## Phase 11 — Building the Interactive Site

**What we did:** Built a 4-page site (Home, Methodology, Validation, Recommendations) to present all of the above clearly — including an interactive pitch trajectory tool, real visual comparisons for the tunneling finding, and coach-usable "game plan" cards built directly from the validated findings for all three real pitchers.

---

## Honest Limitations (carried through to the final site)

- Hard-hit model (AUC 0.546) is the weakest of the four — contact quality has more inherent randomness than swing-and-miss or batted-ball angle.
- Tunneling analysis uses release-point proximity as a proxy for true pitch-flight similarity, since frame-by-frame tracking data wasn't available.
- League-wide baselines are built on one month of 2024 data; a full season would produce more stable numbers.
- Small-sample results are automatically flagged with a confidence label rather than reported at face value.
- Sample "college pitcher" data used early in the project was simulated, not real TrackMan data.
- Hendricks' broader decline isn't fully explained by shape, location, or platoon splits — likely involves pitch recognition or reduced deception, outside this tool's current feature set.
- The Claude reasoning layer is a manual process (model output reasoned over, then written into the site), not a live automated API call — a deliberate scope decision given the project timeline.

---

*Built by Matt Easton, Miami University, as a baseball analytics + AI portfolio project.*
