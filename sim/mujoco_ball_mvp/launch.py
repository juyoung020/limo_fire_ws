"""MuJoCo-only ball-versus-fire-target proof of concept (no real hardware)."""
import argparse
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


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


def nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply MuJoCo-order quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.array((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ))


def quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    qv = quat[1:]
    uv = np.cross(qv, vector)
    uuv = np.cross(qv, uv)
    return vector + 2.0 * (quat[0] * uv + uuv)


def launcher_aim_quats(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yaw = math.radians(args.yaw)
    pitch = math.radians(args.pitch)
    yaw_quat = np.array((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    # With z up and x forward, a negative y-axis rotation raises the muzzle.
    pitch_quat = np.array((math.cos(pitch / 2), 0.0, -math.sin(pitch / 2), 0.0))
    return yaw_quat, pitch_quat, quat_multiply(yaw_quat, pitch_quat)


def launcher_local_to_world(args: argparse.Namespace, local_pos: tuple[float, float, float]) -> np.ndarray:
    yaw_quat, pitch_quat, _ = launcher_aim_quats(args)
    pitch_pivot = np.array((0.0, 0.0, 0.229))
    point = pitch_pivot + quat_rotate(pitch_quat, np.array(local_pos, dtype=float))
    return quat_rotate(yaw_quat, point)


def arguments(argv=None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="MuJoCo ball/target physics MVP")
    p.add_argument("--headless", action="store_true", help="no GUI, useful for test servers")
    p.add_argument("--speed", type=positive, default=5.0, help="initial ball speed in m/s")
    p.add_argument("--pitch", type=angle, default=20.0, help="elevation in degrees")
    p.add_argument("--yaw", type=angle, default=0.0, help="horizontal aim in degrees")
    p.add_argument("--aim-demo", action="store_true", help="smoothly move from a miss to --yaw/--pitch, then fire")
    p.add_argument("--aim-start-yaw", type=angle, default=20.0, help="starting chassis yaw for --aim-demo")
    p.add_argument("--aim-start-pitch", type=angle, default=5.0, help="starting launcher pitch for --aim-demo")
    p.add_argument("--aim-time", type=positive, default=1.2, help="seconds spent visibly aiming")
    p.add_argument("--distance", type=positive, default=1.25, help="target pivot x coordinate in m")
    p.add_argument("--drag-coefficient", type=nonnegative, default=0.0,
                   help="constant unspun-sphere drag coefficient; 0 disables aerodynamic drag")
    p.add_argument("--air-density", type=positive, default=1.2044,
                   help="air density in kg/m^3 used with --drag-coefficient")
    p.add_argument("--duration", type=positive, default=6.0, help="simulation time in seconds")
    p.add_argument("--elastic-launcher", action="store_true",
                   help="use the documented DS3218 load/catch/release sequence and calibrated launch")
    p.add_argument("--launch-delay", type=nonnegative, default=None,
                   help="time from simulation start until the ball leaves the muzzle")
    p.add_argument("--csv", type=Path, help="optional CSV path for time, target angle and ball pose")
    p.add_argument("--summary", type=Path, help="optional machine-readable JSON result")
    p.add_argument("--scene", type=Path, default=ROOT / "scene.xml")
    p.add_argument("--pivot-height", type=positive, help="hinge height above ground in m")
    p.add_argument("--target-mass", type=positive, help="box mass in kg; inertia recomputed by compiler")
    p.add_argument("--damping", type=nonnegative, help="hinge damping in N m s/rad")
    p.add_argument("--frictionloss", type=nonnegative, help="hinge dry friction in N m")
    p.add_argument("--timestep", type=positive, help="integration timestep in s")
    p.add_argument("--auto-close", action="store_true", help="close GUI after simulation and view hold")
    p.add_argument("--view-hold", type=nonnegative, default=5.0, help="final-frame seconds before auto-close")
    p.add_argument("--hide-panels", action="store_true",
                   help="start the GUI with both MuJoCo side panels hidden")
    args = p.parse_args(raw)
    if args.elastic_launcher:
        if "--scene" not in raw:
            args.scene = ROOT / "scene_elastic_launcher.xml"
        if "--speed" not in raw:
            args.speed = 4.20
        if "--pitch" not in raw:
            args.pitch = 15.0
        if args.launch_delay is None:
            args.launch_delay = (args.aim_time + 1.275) if args.aim_demo else 1.275
        if "--duration" not in raw:
            args.duration = (args.launch_delay + 2.0) if args.aim_demo else 12.0
    elif args.launch_delay is None:
        args.launch_delay = 0.0
    return args


def load_model(args):
    # Compile after overrides so mass and inertia remain physically consistent.
    root = ET.parse(args.scene).getroot()
    if args.elastic_launcher:
        aim_yaw, aim_pitch = args.yaw, args.pitch
        if args.aim_demo:
            args.yaw, args.pitch = args.aim_start_yaw, args.aim_start_pitch
        yaw_quat, pitch_quat, _ = launcher_aim_quats(args)
        args.yaw, args.pitch = aim_yaw, aim_pitch
        root.find(".//body[@name='limo_yaw_root']").set("quat", " ".join(map(str, yaw_quat)))
        root.find(".//body[@name='launcher_pitch_root']").set("quat", " ".join(map(str, pitch_quat)))
    body = root.find(".//body[@name='fire_target']")
    if args.pivot_height is not None:
        pos = body.get("pos").split()
        pos[2] = str(args.pivot_height)
        body.set("pos", " ".join(pos))
    if args.target_mass is not None:
        panel = body.find("geom[@name='target_hitbox']")
        counterweight = body.find("geom[@name='target_counterweight']")
        if counterweight is None:
            panel.set("mass", str(args.target_mass))
        else:
            panel_mass = float(panel.get("mass"))
            counterweight_mass = float(counterweight.get("mass"))
            panel_fraction = panel_mass / (panel_mass + counterweight_mass)
            panel.set("mass", str(args.target_mass * panel_fraction))
            counterweight.set("mass", str(args.target_mass * (1.0 - panel_fraction)))
    joint = body.find("joint[@name='target_hinge']")
    for name in ("damping", "frictionloss"):
        if getattr(args, name) is not None:
            joint.set(name, str(getattr(args, name)))
    if args.timestep is not None:
        root.find("option").set("timestep", str(args.timestep))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


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
    aim_yaw, aim_pitch = args.yaw, args.pitch
    if args.aim_demo:
        args.yaw, args.pitch = args.aim_start_yaw, args.aim_start_pitch
    ball_start = (launcher_local_to_world(args, (0.042, 0.0, 0.071))
                  if args.elastic_launcher else np.array((0.042, 0.0, 0.30)))
    args.yaw, args.pitch = aim_yaw, aim_pitch
    data.qpos[q_adr:q_adr + 7] = (*ball_start, 1.0, 0.0, 0.0, 0.0)
    if args.launch_delay <= 0.0 and not args.elastic_launcher:
        data.qvel[v_adr:v_adr + 6] = (*velocity, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    return q_adr, target_qadr, model.geom("ball_geom").id, model.geom("target_hitbox").id


def launcher_phase(t: float, args: argparse.Namespace) -> str:
    if not args.elastic_launcher:
        return "direct_launch"
    if t < 0.12:
        return "load_66deg"
    if t < 0.72:
        return "loaded_wait_0.60s"
    if t < 1.05:
        return "return_to_rest_0deg"
    if t < 1.15:
        return "sear_catch"
    if t < 1.22:
        return "release_lever_minus_18deg"
    if t < args.launch_delay:
        return "elastic_release_lag_0.055s"
    return "projectile_flight"


def animate_launcher(model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace) -> None:
    if not args.elastic_launcher:
        return
    t = float(data.time) - (args.aim_time if args.aim_demo else 0.0)
    if t < 0.0:
        t = 0.0
    lever = 0.0
    plunger = 0.0
    sear = 0.0
    if t < 0.12:
        f = t / 0.12
        lever, plunger, sear = 66.0 * f, -0.030 * f, 10.0 * (1.0 - f)
    elif t < 0.72:
        lever, plunger = 66.0, -0.030
    elif t < 1.05:
        f = (t - 0.72) / 0.33
        lever, plunger = 66.0 * (1.0 - f), -0.030
    elif t < 1.15:
        plunger = -0.030
    elif t < 1.22:
        f = (t - 1.15) / 0.07
        lever, plunger, sear = -18.0 * f, -0.030, 21.0 * f
    elif t < args.launch_delay:
        lever, plunger, sear = -18.0, -0.030 * (1.0 - (t - 1.22) / max(1e-9, args.launch_delay - 1.22)), 21.0
    else:
        lever = -18.0
    _, _, aim_quat = launcher_aim_quats(args)
    for name, angle_deg, pos in (
        ("launcher_lever_visual", lever, (-0.075, 0.034, 0.071)),
        ("launcher_sear_visual", sear, (-0.100, 0.0346, 0.071)),
    ):
        body = model.body(name)
        mid = int(model.body_mocapid[body.id])
        if mid >= 0:
            mechanism_quat = np.empty(4)
            mujoco.mju_axisAngle2Quat(mechanism_quat, np.array([0.0, 1.0, 0.0]), math.radians(angle_deg))
            data.mocap_pos[mid] = launcher_local_to_world(args, pos)
            data.mocap_quat[mid] = quat_multiply(aim_quat, mechanism_quat)
    body = model.body("launcher_plunger_visual")
    mid = int(model.body_mocapid[body.id])
    if mid >= 0:
        data.mocap_pos[mid] = launcher_local_to_world(args, (-0.028 + plunger, 0.0, 0.071))
        data.mocap_quat[mid] = aim_quat


def run(args, quiet=False) -> dict:
    model = load_model(args)
    data = mujoco.MjData(model)
    final_yaw, final_pitch = args.yaw, args.pitch
    ball_qadr, target_qadr, ball_gid, target_gid = setup(model, data, args)
    ball_body_id = model.body("ball").id
    ball_velocity_adr = model.joint("ball_free").dofadr[0]
    ball_area = math.pi * float(model.geom_size[ball_gid, 0]) ** 2
    min_contact_time = None
    topple_started = None
    toppled_at = None
    threshold_rad = math.radians(45)
    hold_seconds = 0.20
    rows = []
    ground_gid = model.geom("ground").id
    ground_contact_time = None
    ball_ground_contact_time = None
    impact_pos = None
    impact_velocity = None
    max_tilt = 0.0
    max_ground_penetration = 0.0
    completed = False
    launched = args.launch_delay <= 0.0

    def step() -> None:
        nonlocal min_contact_time, topple_started, toppled_at, ground_contact_time, launched
        nonlocal impact_pos, impact_velocity, max_tilt, max_ground_penetration
        nonlocal ball_ground_contact_time
        contact_time = float(data.time)
        if args.aim_demo:
            f = min(1.0, data.time / args.aim_time)
            f = f * f * (3.0 - 2.0 * f)
            args.yaw = args.aim_start_yaw + (final_yaw - args.aim_start_yaw) * f
            args.pitch = args.aim_start_pitch + (final_pitch - args.aim_start_pitch) * f
            yaw_quat, pitch_quat, _ = launcher_aim_quats(args)
            model.body_quat[model.body("limo_yaw_root").id] = yaw_quat
            model.body_quat[model.body("launcher_pitch_root").id] = pitch_quat
            if not launched:
                data.qpos[ball_qadr:ball_qadr + 3] = launcher_local_to_world(args, (0.042, 0.0, 0.071))
                data.qvel[ball_velocity_adr:ball_velocity_adr + 6] = 0.0
        animate_launcher(model, data, args)
        if not launched and data.time >= args.launch_delay:
            data.qvel[model.joint("ball_free").dofadr[0]:model.joint("ball_free").dofadr[0] + 6] = (
                args.speed * math.cos(math.radians(args.pitch)) * math.cos(math.radians(args.yaw)),
                args.speed * math.cos(math.radians(args.pitch)) * math.sin(math.radians(args.yaw)),
                args.speed * math.sin(math.radians(args.pitch)), 0.0, 0.0, 0.0,
            )
            launched = True
        data.xfrc_applied[ball_body_id, :] = 0.0
        if not launched:
            data.xfrc_applied[ball_body_id, :3] = -model.body_mass[ball_body_id] * model.opt.gravity
        velocity_before = data.qvel[ball_velocity_adr:ball_velocity_adr + 3].copy()
        if args.drag_coefficient > 0.0:
            speed = float(math.sqrt(velocity_before @ velocity_before))
            data.xfrc_applied[ball_body_id, :3] += (
                -0.5 * args.air_density * args.drag_coefficient * ball_area
                * speed * velocity_before
            )
        mujoco.mj_step(model, data)
        data.xfrc_applied[ball_body_id, :] = 0.0
        ball_contact = False
        ground_contact = False
        for i in range(data.ncon):
            c = data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {ground_gid, ball_gid} and c.dist <= 0:
                if ball_ground_contact_time is None:
                    ball_ground_contact_time = contact_time
            if {int(c.geom1), int(c.geom2)} == {ball_gid, target_gid} and c.dist <= 0:
                ball_contact = True
                if min_contact_time is None:
                    # mj_step contacts describe the beginning of this integration step.
                    min_contact_time = contact_time
                    impact_pos = [float(v) for v in c.pos]
                    impact_velocity = [float(v) for v in velocity_before]
            if {int(c.geom1), int(c.geom2)} == {ground_gid, target_gid} and c.dist <= 0:
                ground_contact = True
                max_ground_penetration = max(max_ground_penetration, -float(c.dist))
                if ground_contact_time is None:
                    ground_contact_time = contact_time
        tilt = abs(float(data.qpos[target_qadr]))
        max_tilt = max(max_tilt, math.degrees(tilt))
        if tilt >= threshold_rad:
            if topple_started is None:
                topple_started = float(data.time)
            elif toppled_at is None and data.time - topple_started >= hold_seconds:
                toppled_at = float(data.time)
        else:
            topple_started = None
        if args.csv:
            pos = data.qpos[ball_qadr:ball_qadr + 3]
            rows.append((float(data.time), *(float(v) for v in pos),
                         math.degrees(float(data.qpos[target_qadr])), contact_time,
                         ball_contact, ground_contact, toppled_at is not None,
                         launcher_phase(float(data.time), args)))

    def finish():
        bid = model.body("fire_target").id
        jid = model.joint("target_hinge").id
        dadr = model.jnt_dofadr[jid]
        result = dict(
            contact=min_contact_time is not None, contact_time_s=min_contact_time,
            toppled=toppled_at is not None, toppled_at_s=toppled_at,
            max_tilt_deg=max_tilt, final_tilt_deg=math.degrees(float(data.qpos[target_qadr])),
            simulated_s=float(data.time), completed=completed,
            termination="duration" if completed else "viewer_closed_early",
            ground_contact_time_s=ground_contact_time,
            ball_ground_contact_time_s=ball_ground_contact_time,
            ground_before_target=(ball_ground_contact_time is not None and min_contact_time is not None
                                  and ball_ground_contact_time <= min_contact_time),
            max_ground_penetration_m=max_ground_penetration,
            impact_position_m=impact_pos, impact_velocity_m_s=impact_velocity,
            ball_final_position_m=data.qpos[ball_qadr:ball_qadr + 3].tolist(),
            ball_final_velocity_m_s=data.qvel[ball_velocity_adr:ball_velocity_adr + 3].tolist(),
            speed_m_s=args.speed, pitch_deg=args.pitch, yaw_deg=args.yaw,
            distance_m=args.distance, duration_s=args.duration,
            drag_coefficient=float(args.drag_coefficient),
            air_density_kg_m3=float(args.air_density), projected_area_m2=ball_area,
            initial_air_drag_N=(0.5 * args.air_density * args.drag_coefficient
                                * ball_area * args.speed ** 2),
            pivot_height_m=float(model.body_pos[bid, 2]), target_mass_kg=float(model.body_mass[bid]),
            principal_inertia_kg_m2=model.body_inertia[bid].tolist(),
            damping=float(model.dof_damping[dadr]), frictionloss=float(model.dof_frictionloss[dadr]),
            timestep_s=float(model.opt.timestep), mujoco_version=mujoco.__version__,
            threshold_deg=45, hold_seconds=0.20,
            launch_delay_s=float(args.launch_delay),
            launcher_sequence=(
                {"lever_load_deg": 66, "load_dwell_s": 0.60, "return_to_rest_s": 0.33,
                 "load_travel_s": 0.12, "catch_dwell_s": 0.10,
                 "release_travel_s": 0.07, "lever_release_deg": -18, "ball_exit_lag_s": 0.055,
                 "release_speed_source": "Notion launcher MuJoCo v4, 15-degree three-shot series",
                 "release_speed_samples_m_s": [4.26, 4.06, 4.28]}
                if args.elastic_launcher else None
            ),
        )
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", newline="", encoding="utf-8") as f:
                out = csv.writer(f)
                out.writerow(("time_s", "ball_x_m", "ball_y_m", "ball_z_m", "target_tilt_deg",
                              "contact_sample_time_s", "ball_target_contact", "target_ground_contact", "toppled",
                              "launcher_phase"))
                out.writerows(rows)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if not quiet:
            print(f"CONTACT: {result['contact']} (first at {min_contact_time} s)")
            print(f"TOPPLED (>=45 deg for 0.20 s): {result['toppled']}")
            print(f"Final target tilt: {result['final_tilt_deg']:.1f} degrees; max: {max_tilt:.3f}")
            print(f"Topple confirmed at: {toppled_at}; completed: {completed} ({result['termination']})")
            if args.csv:
                print(f"CSV: {args.csv.resolve()}")
        return result

    if args.headless:
        while data.time < args.duration:
            step()
        completed = True
    else:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(
            model, data,
            show_left_ui=not args.hide_panels,
            show_right_ui=not args.hide_panels,
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (args.distance / 2, 0, 0.25)
                viewer.cam.distance = (
                    max(2.4, args.distance * 1.70)
                    if args.elastic_launcher else max(2.8, args.distance * 1.8)
                )
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -22
            wall_start = time.perf_counter()
            while viewer.is_running() and data.time < args.duration:
                t0 = time.perf_counter()
                target_time = min(args.duration, t0 - wall_start)
                while data.time < target_time:
                    step()
                viewer.sync()
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - t0)))
            completed = data.time >= args.duration
            # Save and print immediately; holding the view does not extend physics time.
            result = finish()
            hold_start = time.perf_counter()
            while viewer.is_running() and (not args.auto_close or time.perf_counter() - hold_start < args.view_hold):
                viewer.sync()
                time.sleep(1 / 60)
            return result

    return finish()


def main() -> None:
    run(arguments())


if __name__ == "__main__":
    main()
