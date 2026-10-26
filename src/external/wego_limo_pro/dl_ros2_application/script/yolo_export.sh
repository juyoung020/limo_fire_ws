#!/bin/bash
echo ""
echo "================================================"
echo " Starting YOLO model export to TensorRT engine"
echo "================================================"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pt_file_path>"
    exit 1
fi

PT_FILE="$1"
ENGINE_FILE="${PT_FILE%.pt}.engine"

echo "Activating 'dl_env' virtual environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl_env

echo "Exporting YOLO model to TensorRT engine format..."
yolo export model="$PT_FILE" format=engine half=true

if [ ! -f "$ENGINE_FILE" ]; then
    echo "Error: TensorRT engine file was not created. Exiting."
    conda deactivate
    exit 1
fi

echo "TensorRT engine export completed successfully."

conda deactivate

