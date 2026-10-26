#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <h5_file_path>"
    exit 1
fi

H5_FILE="$1"
ONNX_FILE="${H5_FILE%.h5}.onnx"
ENGINE_FILE="${H5_FILE%.h5}.engine"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate dl_env

echo "Converting Keras model to ONNX..."
python3 -c "
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import os
import sys

def convert_keras_to_onnx(h5_path, onnx_path):
    try:
        model = keras.models.load_model(h5_path, compile=False)
    except Exception as e:
        sys.exit(1)

    input_spec = (tf.TensorSpec((None, 110, 300, 3), tf.float32, name='input'),)

    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec, opset=13)
        onnx.save(model_proto, onnx_path)
    except Exception as e:
        sys.exit(1)

    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        sys.exit(1)

if not os.path.exists('$H5_FILE'):
    sys.exit(1)
    
convert_keras_to_onnx('$H5_FILE', '$ONNX_FILE')
"

if [ ! -f "$ONNX_FILE" ]; then
    conda deactivate
    exit 1
fi
echo "ONNX conversion completed successfully."

echo "Building TensorRT engine from ONNX model..."
python3 -c "
import tensorrt as trt
import os
import sys

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_path, engine_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    try:
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                return None
    except Exception as e:
        return None

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    min_shapes = (1, 110, 300, 3)
    opt_shapes = (1, 110, 300, 3)
    max_shapes = (1, 110, 300, 3)
    
    profile.set_shape(input_name, min_shapes, opt_shapes, max_shapes)

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            return None
    except Exception as e:
        return None

    try:
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        return serialized_engine
    except Exception as e:
        return None

if not os.path.exists('$ONNX_FILE'):
    sys.exit(1)
    
build_engine('$ONNX_FILE', '$ENGINE_FILE')
"

echo "TensorRT engine build completed."

if [ ! -f "$ENGINE_FILE" ]; then
    conda deactivate
    exit 1
fi
echo "Optimization complete. Both ONNX and TensorRT engine files are ready."

conda deactivate

