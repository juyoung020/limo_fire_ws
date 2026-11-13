# Wego integrated fire mission — initial pre-fix attempt (2026-11-13)

## What ran

Connected to the existing Wego setup and launched its integrated `limo_fire_printer_fire.launch.xml` in a new, isolated ROS 2 domain 43, Webots port 1236, with the GUI enabled. The already-running domain 42 simulation was left alone. The launch started Webots, Nav2, `fire_manager`, the YOLO detector, `object_mapper`, and the MuJoCo extinguisher action server. No Wego source files were edited.

The runtime ROS graph confirmed these links:

- `/camera/color/image_raw` publisher: `camera`; subscriber: `yolo_detector`.
- `/yolo_detector/detections` publisher: `yolo_detector`; subscribers: `object_mapper`.
- `/fire/observations` publisher: `object_mapper`; subscriber: `fire_manager`.
- `/fire/mujoco_extinguish` action server: present.

## Observed outcome

An initial pose was published at the Webots robot spawn `(-4.8, 0.0)` and a goal was sent through `/fire/mission_navigate_to_pose`. Nav2 moved the robot and replanned during travel. The mission state later read `FIRE_DETECTED|unlocalized_fire`. A sampled `/yolo_detector/detections` message was empty, and the mission goal ended `ABORTED`. A second goal to a near-target stand-off pose also ended `ABORTED`.

No localized `/fire/observations` target, MuJoCo launch, projectile contact, topple, or video recording was confirmed. The already-running domain 42 simulation was not stopped.

## Confirmed cause

The Webots depth sensor used by `limo_sensors.py` has a 0.3–3.0 m valid range. YOLO can detect the target before it enters this range, at which point `object_mapper` publishes its `unlocalized_fire` sentinel. The fire manager queued the sentinel as a real target and then aborted because it had no position. Timestamp synchronization and camera TF were checked as possible causes but were not the cause demonstrated by this run.

The fix and subsequent runs are documented in the [Wego integrated run report](../wego-integrated-2026-11-13/README.md). That report distinguishes the successful shot from a later Nav2 progress failure and links the recorded artifacts.
