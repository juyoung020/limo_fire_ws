"""Repeat the best CAD-ball firing settings with small, seeded setup errors."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import mujoco
from launch import ROOT, arguments, run


POINTS = (
    (0.80, 4.2, 20.0),
    (1.25, 4.2, 30.0),
    (1.75, 5.0, 30.0),
    (2.25, 6.0, 20.0),
    (2.75, 6.0, 30.0),
    (2.90, 6.0, 30.0),
    (3.00, 6.0, 30.0),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments" / "2026-11-09" / "cad-launcher-grid")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--distance-jitter", type=float, default=0.025)
    parser.add_argument("--speed-jitter", type=float, default=0.10)
    parser.add_argument("--pitch-jitter", type=float, default=0.75)
    parser.add_argument("--yaw-jitter", type=float, default=0.5)
    parser.add_argument("--drag-coefficient", type=float, default=0.0)
    parser.add_argument("--air-density", type=float, default=1.2044)
    parser.add_argument("--operating-point", type=float, nargs=3, action="append",
                        metavar=("PIVOT_X_M", "SPEED_M_S", "PITCH_DEG"),
                        help="repeat this option to verify a custom operating-point list")
    cli = parser.parse_args()
    if cli.repeats < 1:
        parser.error("--repeats must be positive")
    if min(cli.distance_jitter, cli.speed_jitter, cli.pitch_jitter, cli.yaw_jitter) < 0:
        parser.error("jitter magnitudes must be nonnegative")
    points = tuple(tuple(p) for p in cli.operating_point) if cli.operating_point else POINTS
    if any(not 0.7 <= x <= 3.0 or speed <= 0 or not -65 <= pitch <= 65
           for x, speed, pitch in points):
        parser.error("operating points require pivot x in [0.7, 3.0], positive speed, and pitch in [-65, 65]")
    rng = random.Random(cli.seed)
    rows = []
    by_distance = defaultdict(list)
    for distance, speed, pitch in points:
        for repeat in range(1, cli.repeats + 1):
            actual = {
                "distance": min(3.0, max(0.7, distance + rng.uniform(-cli.distance_jitter, cli.distance_jitter))),
                "speed": speed + rng.uniform(-cli.speed_jitter, cli.speed_jitter),
                "pitch": pitch + rng.uniform(-cli.pitch_jitter, cli.pitch_jitter),
                "yaw": rng.uniform(-cli.yaw_jitter, cli.yaw_jitter),
            }
            args = arguments([
                "--headless", "--scene", str(ROOT / "scene_cad.xml"), "--duration", "6",
                "--distance", str(actual["distance"]), "--speed", str(actual["speed"]),
                "--pitch", str(actual["pitch"]), "--yaw", str(actual["yaw"]),
                "--drag-coefficient", str(cli.drag_coefficient),
                "--air-density", str(cli.air_density),
            ])
            result = run(args, quiet=True)
            row = {
                "nominal_distance_m": distance, "nominal_speed_m_s": speed,
                "nominal_pitch_deg": pitch, "repeat": repeat,
                "actual_distance_m": actual["distance"], "actual_speed_m_s": actual["speed"],
                "actual_pitch_deg": actual["pitch"], "actual_yaw_deg": actual["yaw"],
                "contact": result["contact"], "contact_time_s": result["contact_time_s"],
                "toppled": result["toppled"], "toppled_at_s": result["toppled_at_s"],
                "max_tilt_deg": result["max_tilt_deg"],
                "impact_velocity_m_s": json.dumps(result["impact_velocity_m_s"]),
                "ground_before_target": result["ground_before_target"],
            }
            rows.append(row)
            by_distance[distance].append(row)

    cli.output.mkdir(parents=True, exist_ok=True)
    with (cli.output / "operating_point_trials.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for distance, speed, pitch in points:
        group = by_distance[distance]
        successes = sum(row["toppled"] for row in group)
        contacts = [row["contact_time_s"] for row in group if row["contact_time_s"] is not None]
        confirms = [row["toppled_at_s"] for row in group if row["toppled_at_s"] is not None]
        summaries.append({
            "distance_m": distance, "speed_m_s": speed, "pitch_deg": pitch,
            "trials": len(group), "contacts": sum(row["contact"] for row in group),
            "topple_successes": successes, "success_rate": successes / len(group),
            "median_contact_time_s": statistics.median(contacts) if contacts else None,
            "median_topple_confirmed_s": statistics.median(confirms) if confirms else None,
            "max_tilt_deg": max(row["max_tilt_deg"] for row in group),
        })
    with (cli.output / "operating_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        "scene": "scene_cad.xml", "mujoco_version": mujoco.__version__,
        "drag_coefficient": cli.drag_coefficient, "air_density_kg_m3": cli.air_density,
        "seed": cli.seed, "repeats_per_point": cli.repeats,
        "jitter": {"distance_m": cli.distance_jitter, "speed_m_s": cli.speed_jitter,
                   "pitch_deg": cli.pitch_jitter, "yaw_deg": cli.yaw_jitter},
        "threshold_deg": 45, "hold_seconds": 0.20,
        "trials": len(rows), "contacts": sum(row["contact"] for row in rows),
        "topple_successes": sum(row["toppled"] for row in rows),
        "success_rate": sum(row["toppled"] for row in rows) / len(rows),
        "points": summaries,
    }
    (cli.output / "operating_points.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
