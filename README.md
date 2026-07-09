# wasrt_ros

A self-contained ROS One (`rospy`) node that runs [WaSR-T](https://github.com/lojzezust/WaSR-T)
(ResNet-101) maritime semantic segmentation on a live camera stream. The model definition and the
sequential inference path are vendored straight into `scripts/wasr_t_node.py` — there is no dependency
on the upstream `wasr_t` package, and nothing is downloaded at runtime. You give it a `.pth` and it runs.

The vendored forward pass was verified bit-for-bit identical to upstream (max abs diff 0.0 across a
sequential run) and strict-loads the published weights with zero key mismatches.

## What it does

- Subscribes to a compressed camera image and the matching `camera_info`.
- Runs sequential WaSR-T inference, keeping the temporal feature buffer alive across frames.
- Publishes a single-channel class-index mask (`mono8`, values 0/1/2), an optional color overlay, and an
  optional `camera_info` scaled to the mask resolution.

Class indices: `0 = obstacle/static`, `1 = water`, `2 = sky`.

## Dependencies

- ROS One with `rospy`, `sensor_msgs`, `cv_bridge`.
- `torch`, `torchvision` (torchvision only supplies the ResNet-101 backbone architecture — no weights are
  downloaded), `numpy`, `opencv`.

On the Orin, install the NVIDIA-provided JetPack wheels for `torch`/`torchvision` rather than the generic
pip ones.

## Build

```
cd ~/catkin_ws/src && cp -r /path/to/wasr_t_ros .
cd ~/catkin_ws && catkin_make && source devel/setup.bash
```

## Run

```
roslaunch wasr_t_ros wasr_t.launch weights:=/path/to/wasrt_mastr1478.pth
```

`fp16` defaults to `true` (recommended on the Orin; auto-falls back to fp32 if there's no CUDA device).
Get the ResNet-101 weights from the upstream releases page (`wasrt_mastr1325.pth` or `wasrt_mastr1478.pth`).

## Topics

| direction | topic | type | notes |
| --- | --- | --- | --- |
| sub | `/mast_cam/compressed` | `sensor_msgs/CompressedImage` | set `compressed:=false` for a raw `sensor_msgs/Image` |
| sub | `/mast_cam/camera_info` | `sensor_msgs/CameraInfo` | cached, scaled, and re-published to match the mask |
| pub | `/mast_cam/wasr_seg` | `sensor_msgs/Image` (`mono8`) | class indices, network resolution by default |
| pub | `/mast_cam/wasr_seg/color` | `sensor_msgs/Image` (`rgb8`) | color overlay, `publish_color:=false` to disable |
| pub | `/mast_cam/wasr_seg/camera_info` | `sensor_msgs/CameraInfo` | intrinsics scaled to mask resolution, `publish_camera_info:=false` to disable |

## Parameters

| param | default | meaning |
| --- | --- | --- |
| `~weights` | (required) | path to the ResNet-101 `.pth` |
| `~fp16` | `true` | half precision (CUDA only) |
| `~size` | `[512, 384]` | network input `[W, H]`; smaller = faster, less accurate |
| `~full_res_output` | `false` | `false` = mask + scaled intrinsics at network resolution; `true` = upsample to full camera resolution |
| `~compressed` | `true` | input is `CompressedImage` vs raw `Image` |
| `~publish_color` | `true` | publish the color overlay topic |
| `~publish_camera_info` | `true` | publish the scaled `camera_info` |
| `~reset_after_s` | `0.0` | clear the temporal buffer after this many seconds with no frames (0 disables) |
| `~stats_interval` | `5.0` | throughput logging interval |

`num_classes` (3) and the temporal context length (`hist_len` = 5) are fixed constants, since that is what
the published weights were trained with.

## Resolution

WaSR-T runs at a fixed network input (default 512x384), and the prediction's true spatial resolution is
~input/4. Upsampling the mask to a full HD frame fabricates detail and costs a larger device->host copy,
a full-res encode per frame, and the matching topic bandwidth. So by default the node publishes at network
resolution and scales `camera_info` to match (`fx, cx` by `W_net/W_cam`, `fy, cy` by `H_net/H_cam`; `D` is
scale-invariant and untouched). This stays geometrically exact even on a non-4:3 feed. Use
`full_res_output:=true` only if a consumer needs a mask aligned pixel-for-pixel with the raw image.

## Real-time behavior and benchmarking

Inference runs in a worker thread; the subscriber callback only stores the newest frame (`queue_size=1`).
The worker always processes the freshest available frame and drops any backlog, so a model slower than the
feed degrades by skipping frames rather than falling behind on stale ones. The temporal buffer then holds
the last 5 *processed* frames, spaced a bit further apart in time, which the model tolerates.

Every `~stats_interval` seconds it logs:

```
in 30.0 Hz | out 24.1 Hz | infer 38.7 ms | drop 20%
```

`in` = camera rate, `out` = mask rate, `infer` = mean inference time, `drop` = fraction of frames skipped.

For more speed beyond fp16, the intended path is exporting the model to TensorRT (e.g. via torch-tensorrt
or an ONNX -> TensorRT engine). The node is plain `nn.Module` inference, so swapping `self.model` for a
compiled/TensorRT module is a localized change.
