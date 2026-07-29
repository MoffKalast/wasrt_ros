#!/usr/bin/env python3
import os
import time
from collections import deque

import numpy as np
import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

SEGMENTATION_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRT_TO_TORCH_DTYPE = {trt.float16: torch.float16, trt.float32: torch.float32, trt.int32: torch.int32}

class EWasrTrtNode:
	def __init__(self):
		rospy.init_node('ewasr_onnx_node')

		engine_path = os.path.expanduser(rospy.get_param('~engine'))
		self.image_topic = rospy.get_param('~image_topic', '/camera/image_cropped')
		seg_topic = rospy.get_param('~seg_topic', '/wasr/image_seg')
		preview_topic = rospy.get_param('~preview_topic', '/wasr/image_preview/compressed')
		self.publish_preview = bool(rospy.get_param('~publish_preview', False))
		self.compressed_in = bool(rospy.get_param('~compressed_input', False))
		self.resize_input = bool(rospy.get_param('~resize_input', True))

		if not torch.cuda.is_available():
			raise RuntimeError('CUDA is not available; check the torch build and that the container sees the GPU')
		self.device = torch.device('cuda')

		self.torch_stream = torch.cuda.Stream()
		self.stream = self.torch_stream.cuda_stream

		# Save logger to instance to prevent garbage collection issues
		self.logger = trt.Logger(trt.Logger.ERROR)
		with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
			self.engine = runtime.deserialize_cuda_engine(f.read())
		if self.engine is None:
			raise RuntimeError('failed to deserialize %s; engines are specific to the GPU and TensorRT version they were built with' % engine_path)

		self.context = self.engine.create_execution_context()

		self.trt = {}
		for i in range(self.engine.num_io_tensors):
			name = self.engine.get_tensor_name(i)
			dtype = self.engine.get_tensor_dtype(name)
			if dtype not in TRT_TO_TORCH_DTYPE:
				raise RuntimeError('unsupported TRT dtype %s on tensor %s' % (dtype, name))

			shape = tuple(self.engine.get_tensor_shape(name))
			if any(d < 0 for d in shape):
				raise RuntimeError('tensor %s has dynamic shape %s; export_ewasr_onnx.py traces a fixed resolution, so rebuild the engine from a static ONNX' % (name, shape))

			tensor = torch.empty(shape, device=self.device, dtype=TRT_TO_TORCH_DTYPE[dtype])
			self.trt[name] = tensor
			self.context.set_tensor_address(name, tensor.data_ptr())

		for name in ('image', 'logits'):
			if name not in self.trt:
				raise RuntimeError('engine has no %r tensor, found %s; it does not look like an eWaSR engine' % (name, sorted(self.trt)))

		# Resolution is baked into the engine at export time, so read it back rather than taking it as
		# a parameter: a mismatch between a launch arg and the engine would otherwise fail silently.
		image_shape = self.trt['image'].shape
		self.size = (int(image_shape[3]), int(image_shape[2]))
		self.dtype = self.trt['image'].dtype

		# Normalization runs on the GPU so we only upload uint8, a quarter of the float32 transfer.
		self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
		self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

		self.bridge = CvBridge()
		self.pub_seg = rospy.Publisher(seg_topic, Image, queue_size=1)
		self.pub_preview = rospy.Publisher(preview_topic, CompressedImage, queue_size=1) if self.publish_preview else None

		self.latencies = deque(maxlen=50)
		self._warmup()

		# Deliberately polling with wait_for_message rather than a Subscriber: this pulls the next frame
		# only once inference has finished, so we never process anything that was sitting in a queue.
		# queue_size=1 does not give the same guarantee, since rospy can hand over a socket-buffered
		# frame that is already stale.
		self.msg_type = CompressedImage if self.compressed_in else Image

		rospy.loginfo('eWaSR TensorRT ready: %dx%d %s, logits %s',
		              self.size[0], self.size[1], str(self.dtype).replace('torch.', ''), tuple(self.trt['logits'].shape))
		rospy.loginfo('engine=%s topic=%s', engine_path, self.image_topic)

	def _warmup(self):
		with torch.cuda.stream(self.torch_stream):
			self.trt['image'].zero_()
			for _ in range(20):
				self.context.execute_async_v3(stream_handle=self.stream)
		self.torch_stream.synchronize()

	def infer(self, bgr):
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

		with torch.cuda.stream(self.torch_stream):
			image = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
			image = (image - self.mean) / self.std
			self.trt['image'].copy_(image)

			self.context.execute_async_v3(stream_handle=self.stream)

			# The decoder emits at 1/4 input resolution. Upsample the logits, not the labels, and argmax
			# afterwards: argmax on the coarse map then nearest-upsampling loses thin obstacles outright.
			logits = F.interpolate(self.trt['logits'].float(), size=(self.size[1], self.size[0]), mode='bilinear', align_corners=False)
			labels = logits.argmax(1).squeeze(0).to(torch.uint8).cpu()

		self.torch_stream.synchronize()

		return labels.numpy()

	def prepare(self, msg):
		if self.compressed_in:
			bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
		else:
			bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		if bgr is None:
			return None
		if bgr.shape[1] != self.size[0] or bgr.shape[0] != self.size[1]:
			if not self.resize_input:
				rospy.logwarn_throttle(5.0, 'input %dx%d != engine %dx%d, skipping',
				                       bgr.shape[1], bgr.shape[0], self.size[0], self.size[1])
				return None
			rospy.logwarn_once('resizing input %dx%d -> %dx%d', bgr.shape[1], bgr.shape[0], self.size[0], self.size[1])
			# Training resizes with PIL's BOX filter, which is area averaging. INTER_AREA matches it;
			# the cv2 default of INTER_LINEAR aliases on downscale and shows up as speckle on thin
			# obstacles near the horizon.
			interp = cv2.INTER_AREA if bgr.shape[1] > self.size[0] else cv2.INTER_LINEAR
			bgr = cv2.resize(bgr, self.size, interpolation=interp)
		return bgr

	def run(self):
		while not rospy.is_shutdown():
			try:
				msg = rospy.wait_for_message(self.image_topic, self.msg_type, timeout=1.0)
			except rospy.ROSException:
				continue

			try:
				bgr = self.prepare(msg)
			except Exception as exc:
				rospy.logerr_throttle(5.0, 'failed to decode image: %s', exc)
				continue
			if bgr is None:
				continue

			age = (rospy.Time.now() - msg.header.stamp).to_sec() if msg.header.stamp != rospy.Time() else 0.0

			t0 = time.time()
			labels = self.infer(bgr)
			self.latencies.append(time.time() - t0)

			self.publish(labels, bgr, msg.header)
			rospy.loginfo_throttle(2.0, 'infer %.1f ms (mean %.1f over %d), frame age %.0f ms',
			                       1000.0 * self.latencies[-1],
			                       1000.0 * sum(self.latencies) / len(self.latencies),
			                       len(self.latencies), 1000.0 * age)

	def publish(self, labels, original_bgr, header):
		seg_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
		seg_msg.header = header
		self.pub_seg.publish(seg_msg)

		if self.pub_preview is None:
			return

		color_bgr = cv2.cvtColor(SEGMENTATION_COLORS[labels], cv2.COLOR_RGB2BGR)
		overlay = cv2.addWeighted(original_bgr, 0.5, color_bgr, 0.5, 0.0)
		label = 'eWaSR TensorRT %dx%d' % self.size
		for colour, thickness in (((0, 0, 0), 2), ((255, 255, 255), 1)):
			cv2.putText(overlay, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thickness, cv2.LINE_AA)
		ok, jpg = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
		if not ok:
			return
		msg = CompressedImage()
		msg.header = header
		msg.format = 'jpeg'
		msg.data = jpg.tobytes()
		self.pub_preview.publish(msg)

def main():
	EWasrTrtNode().run()

if __name__ == '__main__':
	main()
