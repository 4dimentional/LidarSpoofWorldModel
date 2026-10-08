# Stage 1: emergent vulnerability in a nominal-only SLAM world model

This iteration reframes the project from "regress SLAM output" to "build a world
model whose generated future inherits the real system's vulnerability." The attack
is NEVER seen in training; it is applied only at inference, as an input
perturbation. The question is whether the failure EMERGES, as it would in a
faithful physics simulator — not whether it was learned.

## The pivot (and why v1/v2 could not work)

Nominal analysis settled the design. slam_toolbox's scan dependence lives in the
**absolute / accumulated** pose, not the per-step increment: calibrated command
integration (dead reckoning) drifts 0.4–4.8 m over a 60 s run, while slam_toolbox
stays locked to ground truth within ~5 cm. Per step, the command alone already
explains motion to ~1–3 mm, and the scan correction is sub-mm noise. v1 and v2
regressed the per-step increment, so they learned the command shortcut, ignored
the scan, and were structurally non-vulnerable. The attack, which corrupts
accumulated scan-matching, could not reach them.

## The model

An absolute-pose recurrent filter. Each step advances a drifting calibrated
command prior, then applies a **scan-only** correction from a recurrent memory
(the "map" proxy). The correction head never sees the command, so a corrupted scan
mechanically corrupts localization. It is trained by rolling its OWN pose across a
48-step window (scheduled sampling annealing to full self-integration), so without
a scan-driven correction the estimate drifts and the absolute-pose loss is large —
this forces scan dependence, as in real SLAM. Nominal-only; 613 k parameters.

## Result 1 — the attack effect emerges, at roughly the right magnitude

Attack effect = cumulative trajectory divergence the attack causes. For the world
model this is its attacked run vs its own clean-scan counterfactual run (same
trajectory, same controller). For the real system it is the attacked episode's
SLAM vs the matched-seed nominal episode's SLAM, paired by elapsed time.

| Attack effect (mm) | nominal | attack | recovery |
|---|---:|---:|---:|
| Real slam_toolbox | 11.3 | 111.7 | 162.2 |
| **World model (emergent)** | **0.0** | **102.8** | **175.6** |
| Calibrated physics alone | 0.0 | 0.0 | 0.0 |

The nominal-only world model, perturbed only at inference, predicts an attack
effect within ~10–15 % of the real magnitude **and** reproduces the signature that
recovery error exceeds attack error — the persistent map-corruption effect v1/v2
could not produce. Calibrated physics predicts exactly zero, because the scan does
not enter it. On the stated research question — reproduce real closed-loop attack
effects more faithfully than a calibrated simulator alone — this is a clear,
qualified yes, and the vulnerability is emergent, not trained.

## Result 2 — it is not yet a faithful localizer (the caveat)

Two honest limits, both visible in `worldmodel_emergent.png`:

- **Poor absolute tracking.** On clean scans the world model drifts to 70–240 mm
  (nominal test) and 200–440 mm over the attack episode, versus real SLAM's ~6–60 mm.
  It is a weak absolute localizer; the attack *effect* is accurate largely because
  common-mode drift cancels in the attacked-minus-clean difference.
- **Wrong temporal shape.** Real SLAM resists during the attack and corrects in
  discrete pose-graph jumps; the world model drifts continuously. The phase-averaged
  magnitudes align, but the mechanism (loop closure / graph optimisation) is not
  captured by a plain LSTM memory.

## Why, and what is next

Both limits point to the same causes: **three training episodes** cannot teach a
robust absolute scan-matcher, and an LSTM is a weak stand-in for slam_toolbox's
explicit pose graph and map. The direction is validated — absolute-pose formulation
makes the vulnerability emerge — but faithful reproduction needs:

1. **More nominal data** — many more nominal episodes/trajectories so the scan
   matcher generalises in absolute terms (still nominal-only; no attack training).
2. **A stronger map/loop-closure memory** — an explicit accumulated-map state, not
   just an LSTM hidden vector.
3. **Closed loop (Stage 2)** — let the surrogate's pose estimate drive the
   controller inside the loop (software physics, software attack, ray-cast clean
   scans), to test whether the emergent effect holds when the robot reacts to its
   own corrupted estimate — the actual closed-loop fidelity question.

## Artifacts

- `metrics.json` — nominal faithfulness, emergent-vulnerability phase errors, full trajectories.
- `worldmodel_emergent.png` — trajectories, error-over-time, attack-effect bars.
- Checkpoints under `world_model/checkpoints/pilot_v3_worldmodel/`.
- Code: `world_model/scripts/train_worldmodel.py`, `plot_worldmodel.py`.
