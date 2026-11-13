# Wego Nav2 controller trial — 2026-11-13

## Scope

This records a real remote run on Wego's `limo_fire_presentation` Webots world. It is an external navigation experiment related to the project's future search-to-fire flow; it does not change or extend the local MuJoCo MVP. No YOLO/image detection, autonomous frontier exploration, aiming, firing, impact, or target topple was tested in this run.

Wego reported Ubuntu 22.04.5, ROS 2 Humble, and Webots R2025a. The launch was isolated in ROS domain 42 on a separate Webots instance so it would not interfere with an already-running domain-0 simulation. The robot used the world's four-wheel differential-drive model and the compatible installed differential-drive URDF/controller configuration. Wego's installed ROS Python stack had a `rclpy.parameter.get_parameter_value` import mismatch; a temporary compatibility shim under `/tmp` allowed the launch to proceed. No Wego source files were changed.

## Results

| Controller | Goal (map, m) | Result | Elapsed wall time | Notes |
|---|---:|---:|---:|---|
| Regulated Pure Pursuit | (0.04, 2.20) | Succeeded | Not captured | Final observed pose (0.078, 1.930) m; zero recoveries |
| Regulated Pure Pursuit | (1.44, 4.00) | Succeeded | Not captured | Action reported success |
| Regulated Pure Pursuit | (-1.17, 4.70), frontier candidate | Canceled | — | Repeated collision-ahead / failed-progress reports; canceled manually. A later map probe found no remaining frontiers. |
| MPPI | (0.04, 2.20) | Succeeded | 16 s | Final observed odometry (0.171370, 2.024246) m |
| MPPI | (1.44, 4.00) | Succeeded | 16 s | Final observed odometry (1.540541, 3.742880) m |

For the MPPI run, `/controller_server` reported `nav2_mppi_controller::MPPIController` as the active `FollowPath` plugin. The setup used the Humble MPPI controller and its documented differential-drive model/critic configuration, with a 2.8 s prediction horizon (`56 × 0.05 s`), 1,000 sampled trajectories, forward speed up to 0.26 m/s, and angular speed up to 1.0 rad/s. The settings were assembled in `/tmp/codex-nav2-mppi.yaml`; they were not committed as project configuration.

The two MPPI waypoint actions succeeded, but this was not a paired benchmark: only these two MPPI trials have measured wall time, the RPP timing was not retained, and there was no controlled seed/repetition study. Therefore these results do not establish that MPPI is faster or more accurate than RPP. The frontier failure also shows why gain divided by straight-line distance alone is insufficient: candidates need fresh map data and inflated-costmap reachability/clearance checks before dispatch.

The initially launched Wego configuration encountered a missing GUI display, a controller-manager import error, and an URDF/world drive-model mismatch. The successful isolated run used `DISPLAY=:1`, a temporary import shim, and the compatible diff-drive description. These are deployment/setup issues observed in the remote environment, not changes to the local MuJoCo model.

## Reproduction evidence

Per-goal results are in [navigation_trials.csv](navigation_trials.csv). Wall times were recorded for the two MPPI goals only; blank cells mean the value was not captured. The run used the team's Webots presentation world already present on Wego. The controller implementation is Nav2's [Humble MPPI controller](https://github.com/ros-navigation/navigation2/blob/humble/nav2_mppi_controller/README.md).

At report-writing time the SSH key connection timed out, so the remote GUI's current state could not be rechecked or recorded as still running. A desktop screenshot was inspected but was obscured and is intentionally not included as evidence.
