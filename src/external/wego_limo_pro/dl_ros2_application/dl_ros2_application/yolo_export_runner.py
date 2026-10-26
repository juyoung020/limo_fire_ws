import subprocess
import sys
import os
from ament_index_python.packages import get_package_share_directory

def main(args=None):
    package_share_directory = get_package_share_directory('dl_ros2_application')
    
    script_path = os.path.join(package_share_directory, 'script', 'yolo_export.sh')
    
    if not os.path.exists(script_path):
        print(f"Error: Shell script not found at {script_path}")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: ros2 run dl_ros2_application yolo_export_runner <pt_file_path>")
        sys.exit(1)

    pt_file_path = sys.argv[1]

    try:
        print(f"Executing shell script: {script_path} {pt_file_path}")
        subprocess.run([script_path, pt_file_path], check=True)
        print("Script execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing script: {e}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()
