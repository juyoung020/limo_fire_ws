# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Webots plugin `limo_sensors.py`: the default launch now publishes the same sensor interface as the real LIMO (`ros2 launch wego teleop_launch.py`), with the robot's topic names, message types, `frame_id`s, rates and QoS.
- Orbbec DaBai DC1 camera topics as on the robot: color, aligned depth and IR images with `camera_info`, compressed images and point clouds.
- Camera mount TFs `base_link → camera_mount → camera_rotate → camera_link` with the robot's frame names.

### Changed

- `/scan` as on the robot: ±110°, 248 beams, BEST_EFFORT QoS, no return reported as `0.0`.
- `/imu` at 100 Hz; both worlds now run at a 10 ms `basicTimeStep`.
- `/odom` and the `odom → base_link` TF are published by `limo_sensors.py` at 50 Hz, stamped with the exact simulation time.
- AMCL uses `base_link` as its base frame, and the cartographer RViz view subscribes to `/scan` with Best Effort.
- README describes the robot-matched sensor interface.

### Removed

- `limo_odometry` node; its wheel + IMU odometry now runs inside `limo_sensors.py`.
- `base_link → base_footprint` TF, which the real robot does not have.
- Webots-specific sensor topics such as `/scan/point_cloud` and `/LIMO/<device>/...`.

## [0.0.1] - 2026-10-02

### Added

- Webots simulation of the four-wheel differential AgileX LIMO (`LimoFourDiff.proto`) in an apartment world, driven by `webots_ros2_driver` and `ros2_control`.
- SLAM with cartographer (`cartographer_launch.py`) and navigation with Nav2 + AMCL (`nav2_launch.py`) on a saved map of the world, each with its own RViz view.
- Wheel + IMU odometry (`limo_odometry`), taking the heading from the IMU because the wheels slip when the robot turns.
- Mecanum drive (`drive:=mecanum`) with `LimoMecanum.proto`, `mecanum_drive_controller` and holonomic DWB parameters for Nav2, contributed by Geonwoo Nam.
- README with installation and usage instructions and demo GIFs.

[Unreleased]: https://github.com/inmo-jang/webots_ros2_limo/compare/81845ec...main
[0.0.1]: https://github.com/inmo-jang/webots_ros2_limo/commits/81845ec
