"""Run the archived upstream source, then only repair its Python import scope."""
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
original = subprocess.run(
    [sys.executable, str(root / 'launch.py'), '--headless'],
    capture_output=True, text=True,
)
(root / 'original.log').write_text(original.stdout + original.stderr, encoding='utf-8')
print('Unmodified source exit:', original.returncode)
source = (root / 'launch.py').read_text(encoding='utf-8')
source = source.replace('import mujoco.viewer', 'from mujoco import viewer as mj_viewer')
source = source.replace('mujoco.viewer.launch_passive', 'mj_viewer.launch_passive')
patched = root / 'launch_import_fix.py'
patched.write_text(source, encoding='utf-8')
result = subprocess.run(
    [sys.executable, str(patched), '--headless', '--csv', str(root / 'trace.csv')],
    capture_output=True, text=True,
)
(root / 'reproduced.log').write_text(result.stdout + result.stderr, encoding='utf-8')
print(result.stdout + result.stderr)
raise SystemExit(result.returncode)
