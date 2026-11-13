# LIMO visual meshes

Source: [AgileX Robotics `limo_ros2`, `humble` branch](https://github.com/agilexrobotics/limo_ros2/tree/dcc5a8637f83236ed6778af86ad826e46a20cd92/limo_description).

- `limo_base_visual.stl` is a quadric-decimated visual copy of `limo_base.stl` (188,947 triangles, within MuJoCo's mesh face limit). Units are converted from millimeters with MJCF mesh scale `0.001`.
- `limo_wheel.stl` is copied from the same package and uses the same millimeter-to-meter scale.
- The base visual follows `limo_four_diff.xacro`: yaw `90` degrees and origin at the ground footprint after combining the base joint and visual offsets.
- Wheel centers follow the four wheel joint origins in `limo_four_diff.xacro`: x `±0.10 m`, y `±0.065 m`, z `0.05 m`.

These meshes are static visual geometry in the MVP. They do not add LIMO collision shapes or drive dynamics.
