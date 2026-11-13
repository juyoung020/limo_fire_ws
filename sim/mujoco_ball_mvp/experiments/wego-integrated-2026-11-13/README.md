# Wego integrated fire mission — 2026-11-13

## Scope and setup

This was an actual run of Wego's existing Webots + Nav2 + YOLO + object mapper + fire manager + MuJoCo launcher. It did not add a new ROS 2/Nav2 or MPC stack to this local MVP. The launch used Ubuntu 22.04, ROS 2 Humble, Webots R2025a, ROS domain 43, Webots port 1236, and `limo_fire_printer_obstacles.wbt`. The existing domain 42 simulation on port 1234 was left running. The GUI was enabled and RViz disabled.

The launch was `ros2 launch limo_fire_bt limo_fire_printer_fire.launch.xml port:=1236 rviz:=false show_mujoco_gui:=true`; the mission target was map `(2.8, 0.55)` from the world spawn `(-4.8, 0.0)`. The perception graph was verified: camera color images reached YOLO, YOLO detections reached `object_mapper`, and `/fire/observations` reached `fire_manager`.

## Root cause and code change

The Webots depth camera advertises valid distances from 0.3 m to 3.0 m. YOLO can recognize the flame before it enters that range. In that case the mapper publishes its sentinel observation `id=unlocalized_fire`, with `position_valid=false`. The fire manager had queued this sentinel as a real fire; its stop/avoid path then failed because it had no position, aborting the Nav2 mission before a localized observation could arrive.

On Wego branch `codex/wego-ackermann-printer-world`, `src/limo_fire/limo_fire_bt/src/fire_manager.cpp` now ignores only the `unlocalized_fire` sentinel when its position is invalid, logs that it is waiting for localization, and leaves the active Nav2 goal running. It still requires a valid mapped position before danger-zone planning or firing. No confidence threshold, depth limit, impact check, or topple threshold was relaxed. The change was built with `colcon build --packages-select limo_fire_bt`; commit `d3b686a` was pushed to that feature branch, not `main`.

## Results

| Run | Outcome | Evidence |
|---|---|---|
| Before fix | Unlocalized detection led to mission abort; no shot | [Earlier diagnostic](../wego-nav-2026-11-13/integrated-fire-attempt.md) |
| After fix, run 1 | **Fire localized; MuJoCo action succeeded; Nav2 resumed** | [fire manager log](raw/domain43_evidence/fire_manager_1076627_1794568064011.log), [extinguisher log](raw/domain43_evidence/python3_1076629_1794568066249.log), [shot summary](shot_summary.csv), [shot trace](shot.csv) |
| After fix, retry | Nav2 action aborted with `SAFE_STOP|Nav2 action failed`; controller log reported failure to make progress before target localization. No shot was triggered in this retry. | [fire manager log](raw/domain43_evidence/fire_manager_1101392_1794571603042.log), [start-to-abort GUI capture](raw/domain43_evidence/wego_fire_domain43_retry.mp4) |
| After fix, run 3 (no recording) | Direct and alternate stand-off goals stalled at the same crate choke; Nav2 did not reach the fire. No localized detection or shot. | Remote Nav2/fire-manager logs; domain 43 was stopped after the attempt. |
| After fix, run 4 (no recording) | Alternate waypoint `(-1.0, 2.5)` made no measurable progress and emitted no `/cmd_vel`; the attempt was stopped. | Remote pose/controller observations; domain 43 was stopped after the attempt. |
| After fix, run 5 (no recording) | Direct goal was accepted and robot reached the goal area. Fire manager received only an unlocalized fire hint and kept waiting for a valid depth pose. Nav2 repeatedly aborted controller progress; no localized target or shot occurred. A second orientation goal was rejected while the mission was active. | [fire manager log](raw/domain43_evidence/fire_manager_run5.log), [controller log](raw/domain43_evidence/nav2_run5.log); domain 43 was stopped after the attempt. |

The successful shot used the live mapped target at relative `(1.778, 0.911) m`, range `1.997 m`, and the existing launcher parameters: 5.5 m/s, 1.927603 m pivot distance, target Y 0.910692 m, and 0.18 m aim height. The launcher pitch was set to 16.54 degrees; the shot summary measured 16.995 degrees. The real MuJoCo trace recorded launch at 0.618 s, contact at 1.012 s, topple confirmation at 1.424 s, maximum target tilt 114.049 degrees, and final tilt 90.845 degrees. `contact`, `direct_hit`, `toppled`, and `direct_toppled` were all true; failure was `success`. The existing 45-degree-for-0.20-second success rule was unchanged.

One of five post-fix launch attempts completed the localization, firing, and topple flow. The other four failed before localization: three stalled around the obstacle course and one reached the goal area but repeatedly lacked a valid depth pose while Nav2's controller aborted progress. This is not a controlled navigation success-rate benchmark. The successful manager log confirms resumption after the shot; the mission action was not observed returning its final navigation result before the launch was stopped. Navigation/perception reliability therefore remains unresolved.

## GUI recordings

- [Post-shot GUI recording](wego_fire_domain43_screen.mp4) was captured after the successful action had already finished, so it shows the result GUI only and is not a full mission recording. The screen capture was overloaded: 1,378 frames were recorded over 354 seconds but encoded at 10 fps, so do not use its playback duration as a run-time measurement.
- [Retry GUI recording](raw/domain43_evidence/wego_fire_domain43_retry.mp4) started before the repeat launch and records the actual Nav2 retry through its failure. It does not show a successful firing sequence.

No video was captured from initial movement through the successful detection, firing, and topple. The CSVs and logs are the evidence for that successful run; the video limitations are recorded here rather than presenting the failed retry as a successful demo.
