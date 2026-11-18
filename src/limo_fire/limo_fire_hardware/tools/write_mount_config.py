#!/usr/bin/env python3
"""Write measured launcher geometry into a copy of approach_aim.yaml.

Centimetre inputs are measured at the pitch-pivot centre. The template remains
unchanged, and motion/serial stay disabled until the operator launches them.
"""

import argparse
import math
from pathlib import Path
import re


TEMPLATE = Path(__file__).resolve().parents[1] / 'config' / 'approach_aim.yaml'


def _replace(text, key, value):
    pattern = rf'(?m)^(\s*{re.escape(key)}:\s*).*$'
    result, count = re.subn(pattern, lambda match: match.group(1) + value, text)
    if count != 1:
        raise ValueError(f'Expected one {key} parameter in the template, found {count}')
    return result


def make_config(template, x_cm, y_cm, pivot_height_cm, mount_yaw_deg,
                base_link_height_cm=14.5):
    values = (x_cm, y_cm, pivot_height_cm, mount_yaw_deg, base_link_height_cm)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('All measurements must be finite numbers')
    x, y = x_cm / 100.0, y_cm / 100.0
    z = (pivot_height_cm - base_link_height_cm) / 100.0
    if (abs(x) > 0.5 or abs(y) > 0.5 or not 0 < z < 0.8
            or not -180 <= mount_yaw_deg <= 180 or base_link_height_cm <= 0):
        raise ValueError('Measurements are outside the supported launcher mounting envelope')
    result = _replace(template, 'mount_xyz_m', f'[{x:.4f}, {y:.4f}, {z:.4f}]')
    result = _replace(result, 'mount_yaw_deg', f'{mount_yaw_deg:.3f}')
    result = _replace(result, 'mount_calibrated', 'true')
    result = _replace(result, 'motion_enabled', 'false')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--x-cm', type=float, required=True, help='pivot forward of axle midpoint; backward is negative')
    parser.add_argument('--y-cm', type=float, required=True, help='pivot left of body centreline; right is negative')
    parser.add_argument('--pivot-height-cm', type=float, required=True, help='floor to pitch-pivot centre')
    parser.add_argument('--mount-yaw-deg', type=float, required=True, help='horizontal launcher offset from body +x; left is positive')
    parser.add_argument('--base-link-height-cm', type=float, default=14.5,
                        help='floor to base_link origin in the active robot model (default: 14.5)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurements-confirmed', action='store_true',
                        help='confirm all four values were measured on the mounted robot')
    args = parser.parse_args()
    if not args.measurements_confirmed:
        parser.error('Measure the mounted launcher first, then pass --measurements-confirmed')
    if args.output.resolve() == TEMPLATE.resolve():
        parser.error('Choose an output file; the repository template cannot be overwritten')
    try:
        result = make_config(TEMPLATE.read_text(encoding='utf-8'), args.x_cm, args.y_cm,
                             args.pivot_height_cm, args.mount_yaw_deg, args.base_link_height_cm)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding='utf-8')
    print(f'Wrote {args.output.resolve()} (motion_enabled: false)')


if __name__ == '__main__':
    main()
