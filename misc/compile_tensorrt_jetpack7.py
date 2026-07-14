import tensorrt as trt

from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in_path", type=Path, default="~/wasrt_320x96.onnx", help="Input ONNX model (fp32, cast to fp16 before build unless --no_fp16)")
ap.add_argument("--out_path", type=Path, default="~/wasrt_320x96_fp16.engine", help="Output TensorRT engine")
ap.add_argument("--workspace_gb", type=float, default=2.0, help="Workspace memory pool limit in GB")
ap.add_argument("--opt_level", type=int, default=5, help="Builder optimization level 0-5, higher = longer build, better tactics")
ap.add_argument("--timing_cache", type=Path, default="~/wasrt_timing.cache", help="Timing cache to speed up rebuilds")
ap.add_argument("--no_fp16", action="store_true", help="Skip fp16 conversion and build the model as-is (fp32)")
args = ap.parse_args()

onnx_path = args.in_path.expanduser()
engine_output_path = args.out_path.expanduser()
timing_cache_path = args.timing_cache.expanduser()

# TRT 11 removed BuilderFlag.FP16 and all weak-typing flags, networks are strongly typed by default.
# Precision now comes from the ONNX model itself, so we cast the graph to fp16 before building.
if not args.no_fp16:
	import onnx
	from onnxconverter_common import float16

	print("Converting ONNX model to fp16...")
	model = onnx.load(str(onnx_path))
	model = float16.convert_float_to_float16(model, keep_io_types=True)
	onnx_path = onnx_path.with_suffix(".fp16.onnx")
	onnx.save(model, str(onnx_path))
	print(f"Saved fp16 model to {onnx_path}")

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

with open(onnx_path, "rb") as model:
	if not parser.parse(model.read()):
		print("ERROR: Failed to parse the ONNX file.")
		for error in range(parser.num_errors):
			print(parser.get_error(error))
		exit(1)

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
config.builder_optimization_level = args.opt_level

if timing_cache_path.exists():
	cache = config.create_timing_cache(timing_cache_path.read_bytes())
else:
	cache = config.create_timing_cache(b"")
config.set_timing_cache(cache, ignore_mismatch=False)

print("Building TensorRT engine (this may take a while at opt_level %d)..." % args.opt_level)
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
	print("ERROR: Engine build failed.")
	exit(1)

timing_cache_path.write_bytes(memoryview(config.get_timing_cache().serialize()))

with open(engine_output_path, "wb") as f:
	f.write(serialized_engine)

print(f"Success! Engine saved to {engine_output_path}")