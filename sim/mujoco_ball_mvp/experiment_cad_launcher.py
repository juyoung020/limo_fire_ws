"""Repeat a distance/speed/pitch grid using the CAD-sized MuJoCo projectile."""
import argparse
import csv
import itertools
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import mujoco
from launch import ROOT, arguments, run


DISTANCES_M = (0.8, 1.25, 1.75, 2.25, 2.75, 3.0)
SPEEDS_M_S = (3.0, 4.0, 4.2, 5.0, 6.0)
PITCHES_DEG = (0.0, 10.0, 20.0, 30.0, 40.0)
REPEATS = 3
SEED = 20261110


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "2026-11-09" / "cad-launcher-grid",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--drag-coefficient", type=float, default=0.0)
    parser.add_argument("--air-density", type=float, default=1.2044)
    parser.add_argument("--distances", type=float, nargs="+", default=DISTANCES_M)
    parser.add_argument("--speeds", type=float, nargs="+", default=SPEEDS_M_S)
    parser.add_argument("--pitches", type=float, nargs="+", default=PITCHES_DEG)
    cli = parser.parse_args()
    if cli.repeats < 1:
        parser.error("--repeats must be positive")

    rng = random.Random(cli.seed)
    trials = []
    distances, speeds, pitches = tuple(cli.distances), tuple(cli.speeds), tuple(cli.pitches)
    if any(not 0.7 <= value <= 3.0 for value in distances):
        parser.error("distances must be target-pivot x coordinates between 0.7 and 3.0 m")
    if any(value <= 0 for value in speeds):
        parser.error("speeds must be positive")
    if any(not -65 <= value <= 65 for value in pitches):
        parser.error("pitches must be between -65 and 65 degrees")
    conditions = list(itertools.product(distances, speeds, pitches))
    for condition_index, (distance, speed, pitch) in enumerate(conditions, start=1):
        for repeat in range(1, cli.repeats + 1):
            # Small repeat-to-repeat variations model modest setup and launch error.
            actual_distance = max(0.7, min(3.0, distance + rng.uniform(-0.025, 0.025)))
            actual_speed = speed + rng.uniform(-0.10, 0.10)
            actual_pitch = pitch + rng.uniform(-0.75, 0.75)
            actual_yaw = rng.uniform(-0.5, 0.5)
            args = arguments([
                "--headless",
                "--scene", str(ROOT / "scene_cad.xml"),
                "--duration", "6",
                "--distance", str(actual_distance),
                "--speed", str(actual_speed),
                "--pitch", str(actual_pitch),
                "--yaw", str(actual_yaw),
                "--drag-coefficient", str(cli.drag_coefficient),
                "--air-density", str(cli.air_density),
            ])
            result = run(args, quiet=True)
            trials.append({
                "condition_distance_m": distance,
                "condition_speed_m_s": speed,
                "condition_pitch_deg": pitch,
                "repeat": repeat,
                "actual_distance_m": actual_distance,
                "actual_speed_m_s": actual_speed,
                "actual_pitch_deg": actual_pitch,
                "actual_yaw_deg": actual_yaw,
                "contact": result["contact"],
                "contact_time_s": result["contact_time_s"],
                "toppled": result["toppled"],
                "toppled_at_s": result["toppled_at_s"],
                "max_tilt_deg": result["max_tilt_deg"],
                "impact_velocity_m_s": json.dumps(result["impact_velocity_m_s"]),
                "final_tilt_deg": result["final_tilt_deg"],
                "ground_before_target": result["ground_before_target"],
                "completed": result["completed"],
            })
        if condition_index % 10 == 0 or condition_index == len(conditions):
            print(f"completed {condition_index}/{len(conditions)} conditions")

    grouped = defaultdict(list)
    for row in trials:
        key = (row["condition_distance_m"], row["condition_speed_m_s"], row["condition_pitch_deg"])
        grouped[key].append(row)
    rates = []
    for (distance, speed, pitch), rows in grouped.items():
        contact_times = [r["contact_time_s"] for r in rows if r["contact_time_s"] is not None]
        success_times = [r["toppled_at_s"] for r in rows if r["toppled_at_s"] is not None]
        rates.append({
            "distance_m": distance,
            "speed_m_s": speed,
            "pitch_deg": pitch,
            "trials": len(rows),
            "contacts": sum(r["contact"] for r in rows),
            "topple_successes": sum(r["toppled"] for r in rows),
            "direct_topple_successes": sum(r["toppled"] and not r["ground_before_target"] for r in rows),
            "success_rate": sum(r["toppled"] for r in rows) / len(rows),
            "median_contact_time_s": statistics.median(contact_times) if contact_times else "",
            "median_topple_confirmed_s": statistics.median(success_times) if success_times else "",
            "max_tilt_deg": max(r["max_tilt_deg"] for r in rows),
        })

    cli.output.mkdir(parents=True, exist_ok=True)
    write_csv(cli.output / "trials.csv", trials)
    write_csv(cli.output / "success_rates.csv", rates)
    summary = {
        "scene": "scene_cad.xml",
        "mujoco_version": mujoco.__version__,
        "drag_coefficient": cli.drag_coefficient,
        "air_density_kg_m3": cli.air_density,
        "seed": cli.seed,
        "repeat_jitter": {"distance_m": 0.025, "speed_m_s": 0.10, "pitch_deg": 0.75, "yaw_deg": 0.5},
        "threshold_deg": 45,
        "hold_seconds": 0.20,
        "duration_s": 6,
        "conditions": len(conditions),
        "trials": len(trials),
        "contacts": sum(r["contact"] for r in trials),
        "topple_successes": sum(r["toppled"] for r in trials),
        "direct_topple_successes": sum(r["toppled"] and not r["ground_before_target"] for r in trials),
        "success_rate": sum(r["toppled"] for r in trials) / len(trials),
        "condition_grid": {"target_pivot_x_m": distances, "speeds_m_s": speeds, "pitches_deg": pitches},
    }
    (cli.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
