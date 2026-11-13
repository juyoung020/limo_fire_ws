"""Replay the integrated mobile-search shot and MuJoCo physics in the GUI."""
import argparse
import csv
import json
import math
import time
from threading import Event
from pathlib import Path

import mujoco

import launch


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "experiments" / "agile-demo-2026-11-16" / "gui-recording" / "gui-mobile-replay-data"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--slow-motion", type=float, default=0.20,
                        help="playback factor for the recorded flight and topple")
    parser.add_argument("--hold", type=float, default=2.0,
                        help="seconds to hold the final GUI view")
    parser.add_argument("--start-delay", type=float, default=0.0,
                        help="seconds to hold the initial GUI view before playback")
    parser.add_argument("--wait-for-space", action="store_true",
                        help="wait for Space in the GUI before starting the replay")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if not 0 < args.slow_motion <= 1:
        parser.error("--slow-motion must be in (0, 1]")
    if args.hold < 0:
        parser.error("--hold must be nonnegative")
    if args.start_delay < 0:
        parser.error("--start-delay must be nonnegative")

    data_dir = args.data_dir.resolve()
    scene_path = data_dir / "mobile_scene.xml"
    mission = read_csv(data_dir / "mission.csv")
    shot = read_csv(data_dir / "shot.csv")
    events_doc = json.loads((data_dir / "events.json").read_text(encoding="utf-8"))
    shot_result = json.loads((data_dir / "shot_summary.json").read_text(encoding="utf-8"))
    events = {event["event"]: event["time_s"] for event in events_doc["events"]}
    shot_start = float(events["fired"])
    stop_time = float(events["stopped"])
    mission_end = float(mission[-1]["t"])
    flight_end = min(float(shot[-1]["time_s"]), shot_result["toppled_at_s"] + 0.4)
    playback_end = shot_start + flight_end / args.slow_motion + args.hold

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    root_id = model.body("limo_yaw_root").id
    root_mocap = int(model.body_mocapid[root_id])
    ball_qadr = int(model.joint("ball_free").qposadr[0])
    ball_vadr = int(model.joint("ball_free").dofadr[0])
    target_qadr = int(model.joint("target_hinge").qposadr[0])
    target_vadr = int(model.joint("target_hinge").dofadr[0])
    mechanism_ids = {
        name: int(model.body_mocapid[model.body(name).id])
        for name in ("launcher_lever_visual", "launcher_sear_visual", "launcher_plunger_visual")
    }

    launch_args = launch.arguments([
        "--headless", "--elastic-launcher", "--scene", str(scene_path),
        "--speed", "4.2", "--pitch", "15", "--yaw", "0",
        "--drag-coefficient", "0.45", "--air-density", "1.2044",
    ])
    launch_args.aim_demo = False

    mission_times = [float(row["t"]) for row in mission]
    shot_times = [float(row["time_s"]) for row in shot]

    def bracket(times: list[float], value: float) -> tuple[int, int, float]:
        lo = 0
        hi = len(times) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if times[mid] <= value:
                lo = mid
            else:
                hi = mid
        if value <= times[0]:
            return 0, 0, 0.0
        if value >= times[-1]:
            return hi, hi, 0.0
        span = times[hi] - times[lo]
        return lo, hi, 0.0 if span <= 0 else (value - times[lo]) / span

    def update_state(elapsed: float) -> None:
        if elapsed < shot_start:
            mission_t = min(elapsed, mission_end)
            lo, hi, f = bracket(mission_times, mission_t)
            a, b = mission[lo], mission[hi]
            x = float(a["x"]) + (float(b["x"]) - float(a["x"])) * f
            y = float(a["y"]) + (float(b["y"]) - float(a["y"])) * f
            yaw = float(a["yaw"]) + (float(b["yaw"]) - float(a["yaw"])) * f
            pitch = float(a["pitch"]) + (float(b["pitch"]) - float(a["pitch"])) * f
            flight = None
            data.time = max(0.0, mission_t - stop_time)
        else:
            final = mission[-1]
            x, y = float(final["x"]), float(final["y"])
            yaw, pitch = float(final["yaw"]), float(final["pitch"])
            flight = min(flight_end, max(0.0, (elapsed - shot_start) * args.slow_motion))
            lo, hi, f = bracket(shot_times, flight)
            a, b = shot[lo], shot[hi]
            data.time = flight

        launch_args.yaw = math.degrees(yaw)
        launch_args.pitch = math.degrees(pitch)
        yaw_quat, pitch_quat, _ = launch.launcher_aim_quats(launch_args)
        data.mocap_pos[root_mocap] = (x, y, 0.0)
        data.mocap_quat[root_mocap] = yaw_quat
        model.body_quat[model.body("launcher_pitch_root").id] = pitch_quat
        launch.animate_launcher(model, data, launch_args)
        for mocap_id in mechanism_ids.values():
            if mocap_id >= 0:
                data.mocap_pos[mocap_id, :2] += (x, y)

        if flight is None:
            muzzle = launch.launcher_local_to_world(launch_args, (0.042, 0.0, 0.071))
            data.qpos[ball_qadr:ball_qadr + 3] = (x + muzzle[0], y + muzzle[1], muzzle[2])
            data.qvel[ball_vadr:ball_vadr + 6] = 0.0
            data.qpos[target_qadr] = 0.0
            data.qvel[target_vadr] = 0.0
        else:
            data.qpos[ball_qadr:ball_qadr + 3] = tuple(
                float(a[key]) + (float(b[key]) - float(a[key])) * f
                for key in ("ball_x_m", "ball_y_m", "ball_z_m")
            )
            data.qpos[target_qadr] = math.radians(
                float(a["target_tilt_deg"])
                + (float(b["target_tilt_deg"]) - float(a["target_tilt_deg"])) * f
            )
        mujoco.mj_forward(model, data)

    total_frames = 0
    from mujoco import viewer
    start_event = Event()

    def on_key(key: int) -> None:
        if key == 32:  # GLFW_KEY_SPACE
            start_event.set()

    with viewer.launch_passive(model, data, key_callback=on_key,
                               show_left_ui=False, show_right_ui=False) as gui:
        with gui.lock():
            gui.cam.lookat[:] = (0.0, -0.05, 0.24)
            gui.cam.distance = 5.0
            gui.cam.azimuth = 125
            gui.cam.elevation = -28
        update_state(0.0)
        gui.sync()
        while args.wait_for_space and gui.is_running() and not start_event.is_set():
            gui.sync()
            time.sleep(1 / 60)
        delay_start = time.perf_counter()
        while gui.is_running() and time.perf_counter() - delay_start < args.start_delay:
            gui.sync()
            time.sleep(1 / 60)
        start = time.perf_counter()
        while gui.is_running():
            frame_start = time.perf_counter()
            elapsed = frame_start - start
            if elapsed >= playback_end:
                update_state(playback_end)
                gui.sync()
                break
            update_state(elapsed)
            gui.sync()
            total_frames += 1
            time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - frame_start)))

    result = {
        "completed": True,
        "mode": "MuJoCo GUI playback of saved mission path and actual MuJoCo shot trace",
        "mission_duration_s": shot_start,
        "flight_playback_factor": args.slow_motion,
        "initial_view_hold_s": args.start_delay,
        "waited_for_space": args.wait_for_space,
        "gui_playback_duration_s": playback_end,
        "frames": total_frames,
        "detected_at_s": float(events["truth_fov_detected"]),
        "stopped_at_s": stop_time,
        "fire_trigger_at_s": shot_start,
        "ball_release_after_trigger_s": launch_args.launch_delay,
        "ball_release_at_s": shot_start + launch_args.launch_delay,
        "contact_after_trigger_s": shot_result["contact_time_s"],
        "contact_at_s": shot_start + shot_result["contact_time_s"],
        "topple_confirmed_after_trigger_s": shot_result["toppled_at_s"],
        "topple_confirmed_at_s": shot_start + shot_result["toppled_at_s"],
        "toppled": bool(shot_result["toppled"]),
        "topple_rule_deg": 45,
        "topple_hold_s": 0.2,
        "max_tilt_deg": shot_result["max_tilt_deg"],
        "scene_repo_relative": scene_path.relative_to(ROOT).as_posix(),
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
