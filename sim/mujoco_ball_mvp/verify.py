"""Capture regression test results and optionally a real GUI auto-close smoke test."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments' / '2026-11-09')
    args = parser.parse_args()
    # Child processes run in ROOT; keep caller-relative output paths unambiguous.
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    tests = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_physics'],
                           cwd=ROOT, capture_output=True, text=True)
    (args.output / 'tests.log').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    print(tests.stdout + tests.stderr)
    if tests.returncode:
        raise SystemExit(tests.returncode)
    if args.gui:
        start = time.perf_counter()
        result_path = args.output / 'gui_auto_close.json'
        proc = subprocess.run([sys.executable, str(ROOT / 'launch.py'), '--auto-close',
                               '--summary', str(result_path)], cwd=ROOT,
                              capture_output=True, text=True, timeout=45)
        elapsed = time.perf_counter() - start
        (args.output / 'gui_auto_close.log').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        if proc.returncode:
            raise SystemExit(proc.returncode)
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if not result['completed'] or not result['toppled']:
            raise RuntimeError('GUI run did not complete with the expected topple result')
        if elapsed < 11:
            raise RuntimeError('6 simulated seconds + 5 second final view expected')
        record = dict(exit_code=proc.returncode, wall_seconds=elapsed,
                      simulated_s=result['simulated_s'], view_hold_s=5,
                      visual_inspection=False, reason='Automated smoke test does not inspect rendered pixels')
        (args.output / 'gui_verification.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
