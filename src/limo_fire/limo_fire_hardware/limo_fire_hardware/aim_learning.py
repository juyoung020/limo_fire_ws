"""Offline shot log and one-step contextual-bandit preparation.

This module has no ROS imports and no path that publishes motor or fire commands.
The fitted JSON is a candidate artifact; no runtime node loads it.
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 1
FIELDS = (
    'run_id', 'shot_id', 'recorded_at_utc', 'source', 'ball_color',
    'launch_speed_mps', 'target_id', 'range_m', 'height_delta_m',
    'baseline_yaw_deg', 'baseline_pitch_deg', 'delta_yaw_deg',
    'delta_pitch_deg', 'impact_error_right_m', 'impact_error_up_m',
    'contact', 'toppled', 'notes',
)
SOURCES = ('real_camera', 'mujoco')
COLORS = ('blue', 'yellow')


def _number(row, key, *, optional=False):
    raw = row[key].strip()
    if optional and not raw:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f'{key} must be finite')
    return value


def _boolean(row, key):
    value = row[key].strip().lower()
    if value not in ('', 'true', 'false'):
        raise ValueError(f'{key} must be true, false, or blank')
    return None if not value else value == 'true'


def validate_row(row):
    """Validate one manually observed trial; blank impact coordinates are allowed."""
    if set(row) != set(FIELDS):
        raise ValueError('shot CSV columns do not match schema version 1')
    for key in ('run_id', 'shot_id', 'target_id'):
        if not row[key].strip():
            raise ValueError(f'{key} is required')
    if row['source'] not in SOURCES or row['ball_color'] not in COLORS:
        raise ValueError('source or ball_color is unsupported')
    try:
        stamp = datetime.fromisoformat(row['recorded_at_utc'].replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('recorded_at_utc must be ISO 8601') from exc
    if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError('recorded_at_utc must include UTC offset')
    for key in ('launch_speed_mps', 'range_m', 'height_delta_m',
                'baseline_yaw_deg', 'baseline_pitch_deg', 'delta_yaw_deg',
                'delta_pitch_deg'):
        _number(row, key)
    if _number(row, 'launch_speed_mps') <= 0 or _number(row, 'range_m') <= 0:
        raise ValueError('speed and range must be positive')
    right = _number(row, 'impact_error_right_m', optional=True)
    up = _number(row, 'impact_error_up_m', optional=True)
    if (right is None) != (up is None):
        raise ValueError('both impact coordinates must be measured together')
    contact, toppled = _boolean(row, 'contact'), _boolean(row, 'toppled')
    if toppled and contact is False:
        raise ValueError('toppled=true conflicts with contact=false')
    return row


def read_shots(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError('shot CSV header does not match schema version 1')
        seen = set()
        shots = []
        for line, row in enumerate(reader, start=2):
            try:
                validate_row(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f'line {line}: {exc}') from exc
            key = (row['run_id'], row['shot_id'])
            if key in seen:
                raise ValueError(f'line {line}: duplicate run_id/shot_id')
            seen.add(key)
            shots.append(row)
    return shots


def _bin(value, width):
    return math.floor(value / width + 0.5)


def fit_bandit(shots, *, source='real_camera', min_samples=3):
    """Estimate Q(s,a)=-mean impact miss for each observed context/action cell.

    Gamma is zero: each shot is one terminal action. Untried actions have no Q.
    The output is deliberately not an automatic command policy.
    """
    if source not in SOURCES or min_samples < 1:
        raise ValueError('invalid training source or min_samples')
    cells = {}
    used = 0
    for row in shots:
        validate_row(row)
        if row['source'] != source:
            continue
        right = _number(row, 'impact_error_right_m', optional=True)
        up = _number(row, 'impact_error_up_m', optional=True)
        if right is None:
            continue
        key = (row['ball_color'], _bin(_number(row, 'launch_speed_mps'), 0.2),
               _bin(_number(row, 'range_m'), 0.25),
               _bin(_number(row, 'height_delta_m'), 0.10),
               _bin(_number(row, 'baseline_pitch_deg'), 1.0),
               _bin(_number(row, 'delta_yaw_deg'), 0.5),
               _bin(_number(row, 'delta_pitch_deg'), 0.5))
        cells.setdefault(key, []).append(math.hypot(right, up))
        used += 1
    qualified = []
    for (color, speed_bin, range_bin, height_bin, base_pitch_bin,
         yaw_bin, pitch_bin), misses in sorted(cells.items()):
        if len(misses) < min_samples:
            continue
        mean_miss = sum(misses) / len(misses)
        qualified.append(dict(ball_color=color, launch_speed_bin=speed_bin,
                              range_bin=range_bin, height_bin=height_bin,
                              baseline_pitch_bin=base_pitch_bin, delta_yaw_bin=yaw_bin,
                              delta_pitch_bin=pitch_bin, count=len(misses),
                              mean_miss_m=mean_miss, q_value=-mean_miss))
    return dict(schema_version=SCHEMA_VERSION, algorithm='one_step_contextual_bandit',
                source=source, reward='negative_impact_miss_metres',
                launch_speed_bin_mps=0.2, range_bin_m=0.25, height_bin_m=0.10,
                baseline_pitch_bin_deg=1.0, correction_bin_deg=0.5,
                min_samples_per_action=min_samples, measured_shots_used=used,
                cells=qualified, runtime_enabled=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Offline aim learning preparation only')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init').add_argument('csv_path')
    sub.add_parser('validate').add_argument('csv_path')
    fit = sub.add_parser('fit')
    fit.add_argument('csv_path')
    fit.add_argument('model_json')
    fit.add_argument('--source', choices=SOURCES, default='real_camera')
    fit.add_argument('--min-samples', type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == 'init':
        with Path(args.csv_path).open('x', newline='', encoding='utf-8') as stream:
            csv.writer(stream).writerow(FIELDS)
        print(f'Created empty shot log: {args.csv_path}')
        return
    shots = read_shots(args.csv_path)
    if args.command == 'validate':
        print(f'Validated {len(shots)} shot rows')
        return
    model = fit_bandit(shots, source=args.source, min_samples=args.min_samples)
    if not model['cells']:
        raise SystemExit('No context/action cell has enough measured impact samples')
    model['training_csv_sha256'] = hashlib.sha256(Path(args.csv_path).read_bytes()).hexdigest()
    with Path(args.model_json).open('x', encoding='utf-8') as stream:
        json.dump(model, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Wrote offline candidate from {model['measured_shots_used']} measured shots; runtime is disabled")


if __name__ == '__main__':
    main()
