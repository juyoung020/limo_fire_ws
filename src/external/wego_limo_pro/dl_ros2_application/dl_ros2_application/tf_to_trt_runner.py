import subprocess
import sys
import os
from ament_index_python.packages import get_package_share_directory

def main(args=None):
    package_name = 'dl_ros2_application'

    try:
        package_share_directory = get_package_share_directory(package_name)
    except Exception as e:
        print(f"Error finding package share directory: {e}")
        sys.exit(1)

    script_path = os.path.join(package_share_directory, 'script', 'tf_to_trt.sh')

    if len(sys.argv) < 2:
        print("Usage: ros2 run dl_ros2_application tf_to_trt <h5_file_path>")
        sys.exit(1)

    h5_file_path = sys.argv[1]

    try:
        print(f"Executing shell script: {script_path} {h5_file_path}")
        subprocess.run([script_path, h5_file_path], check=True)
        print("Script execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
