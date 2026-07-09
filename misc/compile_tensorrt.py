import onnx
import tensorrt as trt
from onnxconverter_common import float16

onnx_fp16_path = "wasrt_320x96_fp16.onnx"
engine_output_path = "wasrt_320x96_fp16.engine"

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

with open(onnx_fp16_path, "rb") as model:
    if not parser.parse(model.read()):
        print("ERROR: Failed to parse the quantized ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit(1)

config = builder.create_builder_config()

print("Building TensorRT engine (this may take a bit)...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("ERROR: Engine build failed.")
    exit(1)

with open(engine_output_path, "wb") as f:
    f.write(serialized_engine)

print(f"Success! Engine saved to {engine_output_path}")