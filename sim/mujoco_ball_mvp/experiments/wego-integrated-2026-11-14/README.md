# Wego integrated mission run — 2026-11-14

## Result

A Wego/Webots mission ran on ROS domain 43 with the Ackermann MPPI controller. The robot moved from its spawn at approximately `(-4.8, 0.0)` toward a fire observation pose, the mapper localized `mapper_fire_0_0`, and FireManager stopped navigation and invoked the MuJoCo extinguisher.

The MuJoCo result CSV reports `contact=True`, `direct_hit=True`, `toppled=True`, `direct_toppled=True`, and `failure=success`. The configured success expression in `mujoco_extinguisher.py` requires all three of contact, direct hit, and topple. The existing 45° sustained topple check was not modified.

| Measurement | Recorded value |
|---|---:|
| Fire observation | `mapper_fire_0_0` (`SMALL_FIRE`) |
| Distance at observation | 0.8 m (mapper summary, rounded) |
| Launch speed | 5.5 m/s |
| Launch distance | 0.953510 m |
| Configured pitch | 20.0° |
| Recorded pitch | 0.432949875° |
| Contact time | 0.624 s |
| Topple confirmation time | 0.994 s |
| Maximum target tilt | 114.247947° |
| Final target tilt | 76.304028° |
| FireManager result | `simulator confirmed extinguished` |
| Navigation after extinguishing | Resumed; final mission goal result was not captured |

The distinct configured pitch and recorded pitch fields come from the simulator summary; the model uses the launcher mechanism to determine actual trajectory pitch.

## Evidence

- `shot_summary.csv`: one-row MuJoCo shot result.
- `shot.csv`: raw MuJoCo trajectory trace.
- `fire_manager_1203418_1794584163695.log`: Wego FireManager state transitions and depth localization retries.

The FireManager log records two detections without a usable depth pose, then successful mapper localization, `FIRE_DETECTED`, `STOPPING`, `EXTINGUISHING`, verification, and navigation resumption. The backend had earlier returned false `SUCCEEDED` results while the robot was still far from goal. A position check now rejects such results and requires the current map pose to be within 0.30 m of the requested goal.

## Changes and limitations

- Restored `planner_server.use_sim_time: True` in the simulation Nav2 overlay. Without it, the planner's path stamp used system time while Webots TF used simulation time; Nav2 then reported goal success without moving. After the correction, controller feedback showed the robot moving and the fire mission progressed.
- Wego source commit: `60f17e7` (`Reject false-success Nav2 mission results`) on `codex/wego-ackermann-printer-world`. The feature branch update was pushed to its origin remote.
- The integration run verified extinguishing and navigation resumption. The post-extinguishing navigation goal remained active with recovery attempts, so full final-goal completion is not claimed.
- No multi-distance or multi-angle batch has been run in this session; this is one integrated trial.


## GUI recording

`wego-integrated-2026-11-14.mp4` is a 5:59.4, 1280×720, 10 fps H.264 recording captured directly from the Wego X11 desktop during a repeat mission. Webots and MuJoCo were arranged side by side; the launcher settings window was minimized, and Tab followed by Shift+Tab was sent to the MuJoCo viewer to hide its panels. GStreamer demuxed and parsed the completed MP4 successfully.

`fire_manager_1212941_1794585288321.log` is from the recording run. It records depth localization retries, fire detection, extinguishing, `simulator confirmed extinguished`, and navigation resumption. The recording run's temporary MuJoCo output directory was removed when the simulation closed, so the numeric shot metrics above come from the separately archived successful run represented in `shot_summary.csv`; the video run's own numeric shot CSV is not claimed.
