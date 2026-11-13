"""Paired setup-jitter sensitivity study for the constant-Cd drag assumption."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import mujoco
from launch import ROOT, arguments, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments' / '2026-11-09' / 'cad-launcher-grid' / 'drag-sensitivity')
    parser.add_argument('--coefficients', type=float, nargs='+', default=[0.0, 0.2, 0.35, 0.45, 0.6])
    parser.add_argument('--air-density', type=float, default=1.2044)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261115)
    parser.add_argument('--distance-jitter', type=float, default=0.05)
    parser.add_argument('--speed-jitter', type=float, default=0.2)
    parser.add_argument('--pitch-jitter', type=float, default=1.0)
    parser.add_argument('--yaw-jitter', type=float, default=1.0)
    parser.add_argument('--operating-point', type=float, nargs=3, action='append',
                        metavar=('PIVOT_X_M', 'SPEED_M_S', 'PITCH_DEG'),
                        help='repeat for paired operating points; defaults to 1.25m/4.2m/s/30deg')
    cli = parser.parse_args()
    if cli.repeats < 1 or not cli.coefficients or any(c < 0 for c in cli.coefficients):
        parser.error('repeats must be positive and coefficients nonnegative')
    if cli.air_density <= 0 or min(cli.distance_jitter, cli.speed_jitter,
                                    cli.pitch_jitter, cli.yaw_jitter) < 0:
        parser.error('density must be positive and jitter magnitudes nonnegative')
    points = cli.operating_point or [(1.25, 4.2, 30.0)]
    if any(not 0.7 <= x <= 3.0 or speed <= 0 or not -65 <= pitch <= 65
           for x, speed, pitch in points):
        parser.error('operating points require x in [0.7, 3.0], positive speed and pitch in [-65, 65]')

    # Generate perturbations once, then reuse the same realization at every Cd.
    # This paired design isolates drag-model sensitivity from random setup noise.
    rng = random.Random(cli.seed)
    perturbations = {}
    for point in points:
        nominal_x, nominal_speed, nominal_pitch = point
        perturbations[tuple(point)] = [{
            'distance': min(3.0, max(0.7, nominal_x + rng.uniform(-cli.distance_jitter, cli.distance_jitter))),
            'speed': nominal_speed + rng.uniform(-cli.speed_jitter, cli.speed_jitter),
            'pitch': nominal_pitch + rng.uniform(-cli.pitch_jitter, cli.pitch_jitter),
            'yaw': rng.uniform(-cli.yaw_jitter, cli.yaw_jitter),
        } for _ in range(cli.repeats)]

    rows = []
    grouped = defaultdict(list)
    for point in points:
        nominal_x, nominal_speed, nominal_pitch = point
        for repeat, actual in enumerate(perturbations[tuple(point)], start=1):
            for coefficient in cli.coefficients:
                args = arguments([
                    '--headless', '--scene', str(ROOT / 'scene_cad.xml'), '--duration', '6',
                    '--distance', str(actual['distance']), '--speed', str(actual['speed']),
                    '--pitch', str(actual['pitch']), '--yaw', str(actual['yaw']),
                    '--drag-coefficient', str(coefficient), '--air-density', str(cli.air_density),
                ])
                result = run(args, quiet=True)
                row = {
                    'nominal_distance_m': nominal_x, 'nominal_speed_m_s': nominal_speed,
                    'nominal_pitch_deg': nominal_pitch, 'repeat': repeat,
                    'drag_coefficient': coefficient, 'air_density_kg_m3': cli.air_density,
                    'actual_distance_m': actual['distance'], 'actual_speed_m_s': actual['speed'],
                    'actual_pitch_deg': actual['pitch'], 'actual_yaw_deg': actual['yaw'],
                    'contact': result['contact'], 'contact_time_s': result['contact_time_s'],
                    'toppled': result['toppled'], 'toppled_at_s': result['toppled_at_s'],
                    'max_tilt_deg': result['max_tilt_deg'],
                    'impact_velocity_m_s': json.dumps(result['impact_velocity_m_s']),
                }
                rows.append(row)
                grouped[(tuple(point), coefficient)].append(row)

    cli.output.mkdir(parents=True, exist_ok=True)
    with (cli.output / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for point in points:
        for coefficient in cli.coefficients:
            group = grouped[(tuple(point), coefficient)]
            contacts = [r['contact_time_s'] for r in group if r['contact_time_s'] is not None]
            confirms = [r['toppled_at_s'] for r in group if r['toppled_at_s'] is not None]
            successes = sum(r['toppled'] for r in group)
            summaries.append({
                'distance_m': point[0], 'speed_m_s': point[1], 'pitch_deg': point[2],
                'drag_coefficient': coefficient, 'trials': len(group),
                'contacts': sum(r['contact'] for r in group), 'topple_successes': successes,
                'success_rate': successes / len(group),
                'median_contact_time_s': statistics.median(contacts) if contacts else None,
                'median_topple_confirmed_s': statistics.median(confirms) if confirms else None,
                'max_tilt_deg': max(r['max_tilt_deg'] for r in group),
            })
    with (cli.output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    report = {
        'scene': 'scene_cad.xml', 'mujoco_version': mujoco.__version__,
        'air_density_kg_m3': cli.air_density, 'coefficients': cli.coefficients,
        'seed': cli.seed, 'repeats_per_coefficient_and_point': cli.repeats,
        'paired_perturbations': True,
        'jitter': {'distance_m': cli.distance_jitter, 'speed_m_s': cli.speed_jitter,
                   'pitch_deg': cli.pitch_jitter, 'yaw_deg': cli.yaw_jitter},
        'threshold_deg': 45, 'hold_seconds': 0.20,
        'trials': len(rows), 'results': summaries,
    }
    (cli.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
