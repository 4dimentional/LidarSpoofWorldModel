# TurtleBot3 + ROS2 Humble Setup and PULSER Research Framework

This repository contains: 

1. Complete setup and runtime instructions for TurtleBot3 using ROS2 Humble, `slam_toolbox`, and Nav2  
2. A structured overview of the PULSER LiDAR spoofing research framework for evaluating SLAM robustness  

---

# Part I: TurtleBot3 + ROS2 Humble Setup

## 1. Environment Configuration

After installing ROS2 Humble and `slam_toolbox`, configure the runtime environment.

Open your `.bashrc` file:

```bash
nano ~/.bashrc
````

Add the following lines to the bottom:

```bash
source /opt/ros/humble/setup.bash
sudo chmod 666 /dev/ttyUSB0
sudo chmod 666 /dev/ttyACM0

export TURTLEBOT3_MODEL=burger        # change if using waffle or waffle_pi
export ROS_DOMAIN_ID=30               # choose 0–255, must match local machine
export LDS_MODEL=LDS-02
export OPENCR_PORT=/dev/ttyACM0
export OPENCR_MODEL=burger
export ROS_HOSTNAME={TURTLEBOT_IP}
```

Important configuration notes:

* `ROS_DOMAIN_ID` must be identical on the robot and your local machine.
* `TURTLEBOT3_MODEL` must be set on both systems.
* `ROS_HOSTNAME` should match the TurtleBot’s IP address.
* Running `rviz2` directly on the TurtleBot3 can cause instability due to limited compute resources. Always run visualization on your local machine.

Reload the environment:

```bash
source ~/.bashrc
```

---

## 2. Running the Robot Stack

Each component should be launched in a separate terminal.

### Start TurtleBot3 Bringup

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

### Run SLAM (Asynchronous Mapping)

Default launch:

```bash
ros2 launch slam_toolbox online_async_launch.py
```

With explicit parameters:

```bash
ros2 launch slam_toolbox online_async_launch.py \
slam_params_file:=/opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml \
use_sim_time:=false
```

### Launch Navigation (Nav2)

```bash
ros2 launch nav2_bringup navigation_launch.py
```

### Visualization (Run on Local Machine)

```bash
ros2 run rviz2 rviz2
```

### Teleoperation

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```
**Important Note**: Do not run Teleoperation while Nav2 is active. This will create conflicting command publishers to ```/cmd_vel```.


At this point the robot can:

* Build a map using SLAM
* Localize within a map
* Navigate using Nav2
* Be controlled via keyboard teleoperation

---

# Part II: PULSER

## LiDAR Spoofing for Compromising SLAM on ROS2 Using Statistical Beam Adjustment and Temporal Modeling

PULSER is a staged research framework designed to evaluate the resilience of ROS2-based SLAM systems under LiDAR stream manipulation. The workflow aligns with the structure described in the associated research paper.

---

## 1. Paper-Aligned Attack Flow

### Stage A: Environment Discovery

**Paper Reference:** Section IV-A

**Objective**

Identify a ROS2 communication context where an attacker can interact with the robot.

**Script:** `ROSDomainFinder.py`

**Methodology**

* Brute-force enumeration of `ROS_DOMAIN_ID`
* Enumeration of DDS `RMW_IMPLEMENTATION`
* Validation by detecting non-default topics such as `/scan` and `/tf`

The goal is to confirm visibility and interaction capability within the robot’s ROS2 graph.

---

### Stage B: Direct `/scan` Injection Baseline

**Paper Reference:** Sections IV-B and IV-C

**Objective**

Demonstrate that forged `LaserScan` messages can influence SLAM when publish timing dominates the legitimate LiDAR stream.

**Script:** `OriginalScanBuster.py`

**Methodology**

* High-rate republishing
* Direct geometric perturbation of beams

  * Range scaling
  * Yaw offsets
  * Positional offsets

This stage establishes baseline influence capability before stealth refinement.

---

### Stage C: Timing-Constrained Injection

**Paper Reference:** Section IV-D.1, Algorithm 1

**Objective**

Replace naive burst publishing with temporally plausible forwarding aligned to the learned LiDAR cadence.

**Script:** `TemporalScanBuster.py`

**Core Parameters**

* Valid timing window: `0.1 <= dt <= 0.4` seconds
* Exponential Moving Average period tracking with `alpha = 0.2`
* Publish gate aligned to learned scan period
* Burst suppression threshold: `min_period_sec = 0.08`

This stage reduces detectability by matching real sensor timing behavior.

---

### Stage D: Statistical Beam Adjustment

**Paper Reference:** Section IV-D.2, Algorithm 2

**Objective**

Bias scan matching while preserving realistic per-beam statistical structure.

**Script:** `TemporalScanBuster.py`

**Technique**

* Per-beam baseline derived from recent history using median
* Variability estimation using Median Absolute Deviation (MAD)
* Pull toward baseline using `pull = 0.8`
* Global scaling using `bias_scale = 0.9`
* Clipping to physical bounds `range_min` and `range_max`
* Exclusion of the current frame from history baseline to prevent self-reinforcing drift

The result is bounded, low-variance distortion that maintains plausible LiDAR structure.

---

### Stage E: Synthetic Structure Injection / Persistent Map Corruption

**Paper Reference:** Sections V-C and V-D

**Objective**

Demonstrate that pose corruption can produce false environmental structures and persistent deformation.

**Script:** `scan_waller.py`

**Effect**

* Injection of a world-fixed synthetic wall into the LiDAR stream
* Acceptance of virtual geometry by SLAM
* Persistent map deformation

This stage evaluates the downstream semantic impact of scan manipulation.

---

## 2. Script Integration

| Script                  | Function                                                  |
| ----------------------- | --------------------------------------------------------- |
| `ROSDomainFinder.py`    | Discovers usable ROS2 domain and middleware configuration |
| `OriginalScanBuster.py` | Baseline geometric scan manipulation                      |
| `TemporalScanBuster.py` | Timing-aware and statistically constrained manipulation   |
| `scan_waller.py`        | Synthetic structure demonstration                         |

Each script corresponds to a defined phase of the staged methodology.

---

## 3. End-to-End Workflow

1. Discover ROS2 communication parameters using `ROSDomainFinder.py`.
2. Validate scan influence capability using `OriginalScanBuster.py`.
3. Transition to timing-consistent and statistically bounded manipulation using `TemporalScanBuster.py`.
4. Demonstrate structural and semantic impact using `scan_waller.py`.

The progression moves from feasibility validation to stealth refinement and finally to persistent environmental corruption evaluation.
