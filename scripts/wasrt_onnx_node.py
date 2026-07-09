#!/usr/bin/env python3

import time
import rospy
import numpy as np
import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

# Fixed by the published WaSR-T ResNet-101 weights: 3 classes (0 obstacle, 1 water, 2 sky), context length 5.
NUM_CLASSES = 3
HIST_LEN = 5
SIZE = (320, 96)
SEGMENTATION_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class WasrTNode:
	def __init__(self):
		rospy.init_node('wasr_t_node')

		engine_path = rospy.get_param('~engine', '')
		self.image_topic = rospy.get_param('~image_topic', '/mast_cam/compressed')
		self.out_topic = rospy.get_param('~output_topic', '/mast_cam/wasr_seg')

		self.device = torch.device('cuda')
		self.dtype = torch.float16

		self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
		self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

		self.torch_stream = torch.cuda.Stream()
		self.stream = self.torch_stream.cuda_stream

		# Save logger to instance to prevent garbage collection issues
		self.logger = trt.Logger(trt.Logger.ERROR)
		with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
			self.engine = runtime.deserialize_cuda_engine(f.read())

		self.context = self.engine.create_execution_context()
		self.mem = torch.zeros((1, HIST_LEN, 1024, SIZE[1] // 8, SIZE[0] // 8), device=self.device, dtype=self.dtype)

		self.trt = {}
		for i in range(self.engine.num_io_tensors):
			name = self.engine.get_tensor_name(i)
			dtype = self.engine.get_tensor_dtype(name)

			if dtype == trt.float16:
				tdtype = torch.float16
			elif dtype == trt.float32:
				tdtype = torch.float32
			elif dtype == trt.int32:
				tdtype = torch.int32
			else:
				raise RuntimeError(f"Unsupported TRT dtype {dtype}")

			shape = tuple(self.engine.get_tensor_shape(name))
			tensor = torch.empty(shape, device=self.device, dtype=tdtype)
			self.trt[name] = tensor
			self.context.set_tensor_address(name, tensor.data_ptr())

		self.bridge = CvBridge()
		self.pub = rospy.Publisher(self.out_topic, CompressedImage, queue_size=1)

		self._warmup()
		rospy.loginfo('WaSR-T ready: size=%dx%d', SIZE[0], SIZE[1])

	def _warmup(self):
		dummy = torch.zeros((1, 3, SIZE[1], SIZE[0]), device=self.device, dtype=self.dtype)
		self.trt["image"].copy_(dummy)
		self.trt["mem_in"].zero_()

		for _ in range(20):
			self.context.execute_async_v3(stream_handle=self.stream)
		torch.cuda.synchronize()

	def infer(self, bgr):
		resized = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_LINEAR)
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		image = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
		image = (image - self.mean) / self.std

		self.trt["image"].copy_(image)
		self.trt["mem_in"].copy_(self.mem)

		self.context.execute_async_v3(stream_handle=self.stream)
		torch.cuda.synchronize()

		self.mem.copy_(self.trt["mem_out"])
		logits = self.trt["logits"]
		logits = F.interpolate(logits, size=(SIZE[1], SIZE[0]), mode='bilinear', align_corners=False)

		return logits.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()

	def publish(self, labels, bgr, header):
		resized = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_LINEAR)
		# SEGMENTATION_COLORS is RGB; convert to BGR so the jpg encode below writes correct channels.
		mask = cv2.cvtColor(SEGMENTATION_COLORS[labels], cv2.COLOR_RGB2BGR)
		overlay = cv2.addWeighted(mask, 0.6, resized, 0.4, 0.0)

		msg = self.bridge.cv2_to_compressed_imgmsg(overlay)
		msg.header = header
		self.pub.publish(msg)

	def run(self):
		while not rospy.is_shutdown():
			try:
				msg = rospy.wait_for_message(self.image_topic, CompressedImage, timeout=1.0)
			except rospy.ROSException:
				continue

			bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
			if bgr is None:
				continue

			t0 = time.time()
			labels = self.infer(bgr)
			rospy.loginfo('infer %.1f ms', 1000.0 * (time.time() - t0))

			self.publish(labels, bgr, msg.header)

def main():
	WasrTNode().run()

if __name__ == '__main__':
	main()