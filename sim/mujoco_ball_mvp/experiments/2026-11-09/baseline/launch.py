"""MuJoCo-only ball-versus-fire-target proof of concept (no real hardware)."""
import argparse
import csv
import math
import time
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent


def positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def angle(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not -65 <= number <= 65:
        raise argparse.ArgumentTypeError("angle must be between -65 and 65 degrees")
    return number


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MuJoCo ball/target physics MVP")
    p.add_argument("--headless", action="store_true", help="no GUI, useful for test servers")
    p.add_argument("--speed", type=positive, default=5.0, help="initial ball speed in m/s")
    p.add_argument("--pitch", type=angle, default=20.0, help="elevation in degrees")
    p.add_argument("--yaw", type=angle, default=0.0, help="horizontal aim in degrees")
    p.add_argument("--distance", type=positive, default=1.25, help="target pivot x coordinate in m")
    p.add_argument("--duration", type=positive, default=3.0, help="simulation time in seconds")
    p.add_argument("--csv", type=Path, help="optional CSV path for time, target angle and ball pose")
    return p.parse_args()


def setup(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace) -> tuple:
    """Set target position and launch the free ball using its qpos/qvel slices."""
    if not 0.7 <= args.distance <= 3.0:
        raise ValueError("--distance must be 0.7 to 3.0 m for this small scene")
    body_id = model.body("fire_target").id
    model.body_pos[body_id, 0] = args.distance
    q_adr = model.joint("ball_free").qposadr[0]
    v_adr = model.joint("ball_free").dofadr[0]
    target_qadr = model.joint("target_hinge").qposadr[0]

    pitch = math.radians(args.pitch)
    yaw = math.radians(args.yaw)
    speed_xy = args.speed * math.cos(pitch)
    velocity = (speed_xy * math.cos(yaw), speed_xy * math.sin(yaw), args.speed * math.sin(pitch))

    mujoco.mj_resetData(model, data)
    data.qpos[q_adr:q_adr + 7] = (0.05, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0)
    data.qvel[v_adr:v_adr + 6] = (*velocity, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return q_adr, target_qadr, model.geom("ball_geom").id, model.geom("target_hitbox").id


def main() -> None:
    args = arguments()
    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    data = mujoco.MjData(model)
    ball_qadr, target_qadr, ball_gid, target_gid = setup(model, data, args)
    min_contact_time = None
    topple_started = None
    toppled_at = None
    threshold_rad = math.radians(45)
    hold_seconds = 0.20
    rows = []

    def step() -> None:
        nonlocal min_contact_time, topple_started, toppled_at
        mujoco.mj_step(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {ball_gid, target_gid} and c.dist <= 0:
                if min_contact_time is None:
                    min_contact_time = float(data.time)
                break
        tilt = abs(float(data.qpos[target_qadr]))
        if tilt >= threshold_rad:
            if topple_started is None:
                topple_started = float(data.time)
            elif toppled_at is None and data.time - topple_started >= hold_seconds:
                toppled_at = float(data.time)
        else:
            topple_started = None
        if args.csv:
            pos = data.qpos[ball_qadr:ball_qadr + 3]
            rows.append((round(float(data.time), 4), *(round(float(v), 5) for v in pos), round(math.degrees(float(data.qpos[target_qadr])), 3)))

    if args.headless:
        while data.time < args.duration:
            step()
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < args.duration:
                t0 = time.perf_counter()
                step()
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - t0)))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            out = csv.writer(f)
            out.writerow(("time_s", "ball_x_m", "ball_y_m", "ball_z_m", "target_tilt_deg"))
            out.writerows(rows)
    print(f"CONTACT: {min_contact_time is not None} (first at {min_contact_time if min_contact_time is not None else 'none'} s)")
    print(f"TOPPLED (>=45 deg for 0.20 s): {toppled_at is not None}")
    print(f"Final target tilt: {math.degrees(float(data.qpos[target_qadr])):.1f} degrees")
    if toppled_at is not None:
        print(f"Topple confirmed at: {toppled_at:.3f} s")
    if args.csv:
        print(f"CSV: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
