"""Reproducible headless ablations, launch grid, perturbations and timestep checks."""
import csv
import itertools
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

from launch import ROOT, arguments, run


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments' / '2026-11-09')
    cli = parser.parse_args()
    output = cli.output
    output.mkdir(parents=True, exist_ok=True)
    log = []

    def trial(group, label, repeat=0, trace=False, **overrides):
        args = arguments(['--headless'])
        for key, value in overrides.items():
            setattr(args, key, value)
        if trace:
            args.csv = output / 'traces' / f'{group}_{label}.csv'
        result = dict(group=group, case=label, repeat=repeat, **run(args, quiet=True))
        log.append(f"{group}/{label}/{repeat}: contact={result['contact']} toppled={result['toppled']} "
                   f"max={result['max_tilt_deg']:.6f} t={result['toppled_at_s']}")
        return result

    ablations = []
    for pivot in (0.02, 0.08):
        base = dict(pivot_height=pivot)
        ablations.append(trial('ablation', f'pivot_{pivot}_default', trace=True, **base))
        for name, values in dict(speed=(3, 4, 6, 8), pitch=(0, 10, 30, 40),
                                 target_mass=(0.07, 0.28), damping=(0, 0.08),
                                 frictionloss=(0, 0.01)).items():
            for value in values:
                ablations.append(trial('ablation', f'pivot_{pivot}_{name}_{value}',
                                       **base, **{name: value}))
    for pivot in (0.02, 0.04, 0.06, 0.075, 0.08, 0.10):
        ablations.append(trial('clearance', str(pivot), pivot_height=pivot))
    write_csv(output / 'ablations.csv', ablations)

    grid = []
    rates = []
    for distance, speed, pitch in itertools.product((0.8, 1.25, 1.75, 2.25), (4, 5, 6), (0, 10, 20, 30, 40)):
        case = f'd{distance}_v{speed}_p{pitch}'
        results = [trial('grid', case, repeat=i, distance=distance, speed=speed, pitch=pitch)
                   for i in range(3)]
        grid.extend(results)
        rates.append(dict(distance_m=distance, speed_m_s=speed, pitch_deg=pitch, trials=3,
                          contacts=sum(r['contact'] for r in results),
                          successes=sum(r['toppled'] for r in results),
                          direct_successes=sum(r['toppled'] and not r['ground_before_target'] for r in results),
                          ground_first_successes=sum(r['toppled'] and r['ground_before_target'] for r in results),
                          success_rate=sum(r['toppled'] for r in results) / 3))
    write_csv(output / 'grid.csv', grid)
    write_csv(output / 'success_rates.csv', rates)

    rng = random.Random(20261109)
    perturbations = [trial('perturbation', str(i), repeat=i,
                           speed=5 + rng.uniform(-0.25, 0.25),
                           pitch=20 + rng.uniform(-2, 2), yaw=rng.uniform(-1, 1),
                           distance=1.25 + rng.uniform(-0.05, 0.05)) for i in range(100)]
    write_csv(output / 'perturbations.csv', perturbations)
    checks = [trial('timestep', str(dt), timestep=dt, trace=True) for dt in (0.002, 0.001, 0.0005)]
    checks += [trial('control', 'no_launch', speed=0, trace=True),
               trial('control', 'miss', yaw=65, trace=True),
               trial('control', 'short_duration', duration=0.3, trace=True)]
    write_csv(output / 'checks.csv', checks)
    environment = dict(python=sys.version, platform=platform.platform(), seed=20261109,
                       baseline_commit='6cfdb08d5164c694605235083d3f11b7942d80de',
                       grid_trials=len(grid), grid_successes=sum(r['toppled'] for r in grid),
                       grid_direct_successes=sum(r['toppled'] and not r['ground_before_target'] for r in grid),
                       grid_ground_first_successes=sum(r['toppled'] and r['ground_before_target'] for r in grid),
                       perturbation_trials=len(perturbations),
                       perturbation_successes=sum(r['toppled'] for r in perturbations))
    (output / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (output / 'requirements-lock.txt').write_text(freeze, encoding='utf-8')
    (output / 'runs.log').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print(json.dumps(environment, indent=2))


if __name__ == '__main__':
    main()
