# Surrogate SLAM world model

Research question: can a physics-based simulator augmented with a learned,
history-dependent response model reproduce real closed-loop attack effects more
faithfully than a calibrated simulator alone?

This implementation is the first controlled simulation pilot. It does NOT answer
the physical-robot fidelity question. Training targets come from running
slam_toolbox inside Gazebo; it is necessary to add paired real-robot experiments
and a calibrated simulation-only reference before making that claim.

## Goal: emergent vulnerability, not learned attack response

The world model is trained on NOMINAL data only. The attack is never seen in
training; it is applied only at inference, as an input perturbation to the delivered
LiDAR. The question is whether the real system's failure EMERGES in the generated
future, as it would in a faithful physics simulator -- not whether it was learned.
Training on attack data would be circular and is excluded.

## Model: absolute-pose recurrent SLAM transition

`train_worldmodel.py` learns the SLAM transition module -- a surrogate of
slam_toolbox's scan -> pose-estimate map. Nominal analysis of this data showed that
SLAM's scan dependence lives in the ABSOLUTE / accumulated pose, not the per-step
increment: calibrated command integration (dead reckoning) drifts 0.4-4.8 m over a
60 s run while slam_toolbox stays locked to ground truth within ~5 cm, yet per-step
increments are command-explainable to ~1-3 mm. A model that regresses the per-step
increment therefore learns a command shortcut, ignores the scan, and is not
vulnerable to scan attacks. (An earlier MLP/LSTM increment regressor confirmed this:
it reproduced <0.2 mm of the ~29 mm attack response.)

The model instead maintains an absolute pose estimate and a recurrent memory (a map
proxy). Each step advances a drifting calibrated command prior, then applies a
SCAN-ONLY correction from the memory; the correction head never sees the command, so
a corrupted scan mechanically corrupts localization. It is trained by rolling its own
pose across a 48-step window (scheduled sampling annealing to full self-integration),
so without a scan-driven correction the estimate drifts and the absolute-pose loss is
large -- forcing scan dependence as in real SLAM. A parsimony penalty keeps the
correction near zero unless the scan demands otherwise.

- Prior: differential-drive arc integration of the command. Two motion gains are
  calibrated exclusively from independent Gazebo ground truth in nominal TRAINING
  episodes. This is NOT a complete calibration of Gazebo.
- Baseline: calibrated physics alone (prior only, scan ignored) -- the "calibrated
  simulator" half of the research question. It predicts zero attack effect by
  construction, because the scan does not enter it.
- Objective: absolute pose MSE (position + wrapped yaw) over the window + correction
  parsimony. Adam, lr 1e-3 with cosine decay, batch 32, 60 epochs. Best epoch is
  selected only on the nominal validation episode.

## Data and attack

`start_pilot.sh` creates a separate headless Docker container on ROS domain 74 with
localhost-only ROS communication and Docker bridge networking. Each episode starts
fresh Gazebo and slam_toolbox processes at the same robot spawn position. A bounded
heading controller consumes SLAM yaw and uses delivered LiDAR for obstacle stopping.
It drives only the simulated robot. Seeds vary the heading/speed schedule.

Six 60-second episodes: three nominal training, one nominal validation, one nominal
test, one matched-seed attack test. Test data is never used for optimization or
checkpoint selection. Failed preliminary runs are retained but excluded from the
successful run's split manifest.

The attack runs the repository's `src/TemporalScanBuster.py` WITHOUT editing its
algorithm. Both `/scan` endpoints are remapped to `/scan_slam`, preserving the
original same-topic subscription and republishing competition. The clean sensor's
`/scan` is relayed unchanged to `/scan_slam` in BOTH nominal and attack episodes.
The attack retains its source defaults: five-frame history, temporal gate, EMA
alpha 0.2, minimum period 0.08 s, median/MAD adjustment, pull 0.8, and bias scale 0.9.
The author's default node clock remains unchanged. Attack runs from simulation
second 20 to 40, followed by recovery. These trajectory/window choices are OUR
pilot protocol, not a claim to reproduce every paper condition or result.

Source: https://github.com/SPIRE-GMU/PULSER
Author manuscript: https://www.finnsec.dev/publications/pulser-at-ieee-host-2026/

## Provenance and independent truth

Gazebo's `libgazebo_ros_state.so` publishes independent world poses. Ground truth
samples are latest-received model states at the clean scan callback, with receipt
simulation time, because ModelStates has no header. Report the sample-time mismatch;
do not describe this as exact acquisition-time ground truth. Evaluation aligns one
initial valid SLAM/ground-truth pose in SE(2); it does not fit away trajectory drift.

A passive C++ subscriber obtains DDS publisher GIDs, identifies the sensor relay
versus `_scan_interceptor`, and mirrors scans to `/observed/clean_scan`,
`/observed/pulser_scan`, or `/observed/unknown_scan`. Publisher IDs and header stamps
are logged on `/experiment/provenance`. The original attack publisher is not wrapped
or altered. `/scan_slam` recording uses BEST_EFFORT to capture both publisher QoS types.

The bags preserve the complete competing message stream. The initial model dataset
uses one row per clean sensor scan, selecting the monitor's last observed message
with the same header timestamp. This is a measurement convention, NOT proof of
which duplicate slam_toolbox accepted. Accepted-scan instrumentation/event-driven
modeling is still needed for precise middleware-level fidelity. Monitor and bag
traffic also add overhead; timing equivalence to an uninstrumented run is not yet
established. Provenance of publications is resolved, SLAM acceptance is not.

## Run

From the host repository (Docker required):

```bash
bash world_model/scripts/start_pilot.sh pilot_new   # collect episodes (Docker)
python3 world_model/scripts/train_worldmodel.py \
  --manifest world_model/configs/pilot_v3_split.json \
  --output world_model/results/pilot_new_worldmodel --device cuda:0
python3 world_model/scripts/plot_worldmodel.py
```

Choose a GPU assigned to you (`cuda:0` worked in this session), or `--device cpu`.
The training environment requires PyTorch, NumPy, and Matplotlib for plotting. The
ROS environment comes from the bundled image. Operational values reside in the CLI
defaults and the experiment runner.

Checkpoints go under `world_model/checkpoints/<run>/`; metrics and plots under
`world_model/results/<run>/`. Episodes and raw bags are excluded from Git.

## What the measurements establish

1. Nominal faithfulness: does the filter localize (kill the metre-scale command
   drift) on the held-out nominal test episode, as real SLAM does?
2. Emergent vulnerability: the nominal-only model is run on the attack episode's
   corrupted scans and on its paired clean scans; the attack effect it predicts is
   the divergence between the two, compared to the real system's attacked-vs-nominal
   divergence and to calibrated physics (which predicts zero).
3. The attack effect emerges at roughly the right magnitude and reproduces the
   recovery > attack persistence, but absolute localization is not yet faithful and
   the discrete pose-graph correction shape is not captured -- see the run REPORT.

The model is trained on nominal data only; emergence of the attack effect is the
result of interest, not a learned response. Six short episodes and one seed cannot
establish statistical significance, generalization across environments, or
physical-robot fidelity. The next study needs more nominal episodes, an explicit
map / loop-closure memory, a controller driven by the surrogate inside the
simulation loop (closed loop), and eventually paired real-robot reference runs.
