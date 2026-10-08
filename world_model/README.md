# World-model data extraction

`extract_bag.py` reads a completed ROS 2 bag offline. It does not replay the bag,
publish commands, move the robot, or run attacks. ROS Humble Python libraries and
NumPy are already present in `turtlebot3-sim`; no live ROS graph is required.

From the host repository directory:

```bash
docker start turtlebot3-sim
docker exec --user "$(id -u):$(id -g)" \
  -e ROS_LOG_DIR=/tmp/world-model-extract-logs turtlebot3-sim \
  bash --noprofile --norc -c '
    source /opt/ros/humble/setup.bash
    python3 /workspace/world_model/scripts/extract_bag.py \
      /workspace/world_model/data/raw/normal_static_smoke_001 \
      --output /workspace/world_model/data/processed/normal_static_smoke_001
  '
```

Existing output folders are never overwritten. Use another output name for a new
extraction. The extractor defaults to SQLite3 storage and the recorded topics
`/scan`, `/tf`, `/tf_static`, `/clock`, `/odom`, `/cmd_vel`.

## Outputs

Each episode produces `observations.npz` and `summary.json`. Load arrays with
`np.load(path, allow_pickle=False)` inside the container or a NumPy environment.
No fixed 726-dimensional feature vector or training split is constructed yet.

| Array | Meaning |
| --- | --- |
| `scan_ranges` | N × beams original float ranges, including inf/NaN |
| `scan_normalized`, `scan_valid` | N × beams normalized ranges and validity masks |
| `scan_stamp_ns` | int64 sensor acquisition times in simulation nanoseconds |
| `scan_bag_time_ns` | int64 recorder timestamps; system time for this smoke bag |
| `scan_range_min/max`, `scan_angle_min/increment`, `scan_time_increment`, `scan_frame` | Per-scan sensor metadata |
| `slam_pose_xyyaw`, `slam_pose_valid` | N × 3 pose at scan time in map frame; NaN when unavailable |
| `next_slam_delta_xyyaw`, `next_slam_delta_valid` | Next-minus-current map-frame position and wrapped yaw delta; final row invalid |
| `command_vw`, `command_observed`, `command_age_ns` | Last received linear.x/angular.z command, validity, and age; never inferred as zero |
| `raw_command_*` | All original command receipt times and v/w values |
| `tf_*` | Raw transforms: header/bag timestamps, static flag, parent/child frames, xyz + xyzw quaternion |
| `odom_*` | Raw odometry times, pose, planar velocity, parent/child frames |
| `clock_bag_time_ns`, `clock_sim_ns` | Recorded clock message pairs |

TF2 composes and interpolates transforms at each scan's acquisition timestamp.
Leading `/` characters in TF frame IDs are removed for compatibility. Missing or
out-of-range TF is explicitly masked; the extractor never substitutes the latest
pose. Epoch-zero scans are masked because TF2 interprets time zero as latest.
Full-episode offline interpolation may use transforms received later; this is
an offline reconstruction, not a claim of causal online SLAM availability.

`/cmd_vel` is unstamped Twist, so alignment uses bag timestamps: the latest
command received no later than the scan's bag timestamp. `--command-max-age`
(default 0.5 seconds in the bag clock domain) bounds this approximation. Missing
or stale commands yield NaN and a false mask; absence means unobserved, not zero.
`command_age_ns=-1` means no earlier command; a stale command retains its age.

Keep simulation times and bag receipt times separate. Their difference is not
sensor latency because their epochs differ in this recording. Range normalization
uses each scan's declared limits. Invalid normalized values are zero placeholders;
always use `scan_valid` in features and losses.

One bag is treated as one episode. Clock resets, non-increasing scan acquisition
times, or changing beam geometry cause an error instead of silently joining
incompatible sequences. Explicit reset/episode events remain necessary for future
pilots, including resets that do not change the clock. The initial implementation
loads the selected topics into memory and is intended for short pilot bags.

## Verification

```bash
docker exec turtlebot3-sim bash --noprofile --norc -c '
  source /opt/ros/humble/setup.bash
  cd /workspace/world_model/scripts
  PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_extract_bag
'
```

The tests cover timestamp interpolation, missing TF, invalid ranges, no/future/
stale commands, reset detection, and refusing output overwrites.

The normal static smoke episode yields 597 × 360 scan values, 597 timestamp-aligned
poses, and zero observed commands. This verifies extraction, not predictive quality.
Ground truth is not yet defined, and `/odom` is not labeled ground truth. Source
provenance cannot be inferred from same-topic scan republishing. Both ground truth
and scan provenance must be established before the first attack-data collection.
