#!/usr/bin/env python3
import os
import time
import rospy
import numpy as np
import cv2
import tensorrt as trt
import torch
import torch.nn.functional as F

from sensor_msgs.msg import CompressedImage, Image
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
		self.image_topic = rospy.get_param('~image_topic', '/camera/image_cropped')
		seg_topic = rospy.get_param('~seg_topic', '/wasr/image_seg')
		preview_topic = rospy.get_param('~preview_topic', '/wasr/image_preview/compressed')
		self.publish_preview = bool(rospy.get_param('~publish_preview', False))

		self.device = torch.device('cuda')

		self.torch_stream = torch.cuda.Stream()
		self.stream = self.torch_stream.cuda_stream

		# Save logger to instance to prevent garbage collection issues
		self.logger = trt.Logger(trt.Logger.ERROR)
		with open(os.path.expanduser(engine_path), "rb") as f, trt.Runtime(self.logger) as runtime:
			self.engine = runtime.deserialize_cuda_engine(f.read())

		self.context = self.engine.create_execution_context()

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

		# ping-pong buffers: mem_out of frame N is mem_in of frame N+1 via address swap, no copy needed
		self.mem_alt = torch.zeros_like(self.trt["mem_in"])
		self.trt["mem_in"].zero_()

		self.dtype = self.trt["image"].dtype
		self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
		self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

		self.bridge = CvBridge()
		self.pub_seg = rospy.Publisher(seg_topic, Image, queue_size=1)
		self.pub_preview = rospy.Publisher(preview_topic, CompressedImage, queue_size=1) if self.publish_preview else None

		self._warmup()
		rospy.loginfo('WaSR-T ready: size=%dx%d', SIZE[0], SIZE[1])

	def _swap_mem(self):
		self.trt["mem_in"], self.mem_alt = self.mem_alt, self.trt["mem_in"]
		self.context.set_tensor_address("mem_in", self.trt["mem_in"].data_ptr())
		self.context.set_tensor_address("mem_out", self.mem_alt.data_ptr())

	def _warmup(self):
		with torch.cuda.stream(self.torch_stream):
			self.trt["image"].zero_()
			self.trt["mem_in"].zero_()
			for _ in range(20):
				self.context.execute_async_v3(stream_handle=self.stream)
		self.torch_stream.synchronize()
		self.trt["mem_in"].zero_()

	def infer(self, bgr):
		resized = cv2.resize(bgr, SIZE, interpolation=cv2.INTER_LINEAR)
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		with torch.cuda.stream(self.torch_stream):
			image = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
			image = (image - self.mean) / self.std
			self.trt["image"].copy_(image)

			self.context.execute_async_v3(stream_handle=self.stream)

			# mem_alt now holds mem_out; it becomes mem_in on the next frame via _swap_mem
			logits = F.interpolate(self.trt["logits"], size=(SIZE[1], SIZE[0]), mode='bilinear', align_corners=False)
			labels = logits.argmax(1).squeeze(0).to(torch.uint8).cpu()

		self.torch_stream.synchronize()
		self._swap_mem()

		return labels.numpy()

	def publish(self, labels, bgr, header):
		# labels are at network resolution; bring them back to the input frame size (nearest keeps class ids intact)
		if labels.shape[0] != bgr.shape[0] or labels.shape[1] != bgr.shape[1]:
			labels = cv2.resize(labels, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

		seg_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
		seg_msg.header = header
		self.pub_seg.publish(seg_msg)

		if self.pub_preview is not None:
			# SEGMENTATION_COLORS is RGB; convert to BGR so the jpg encode below writes correct channels.
			mask = cv2.cvtColor(SEGMENTATION_COLORS[labels], cv2.COLOR_RGB2BGR)
			overlay = cv2.addWeighted(bgr, 0.5, mask, 0.5, 0.0)
			msg = self.bridge.cv2_to_compressed_imgmsg(overlay, dst_format='jpg')
			msg.header = header
			self.pub_preview.publish(msg)

	def run(self):
		while not rospy.is_shutdown():
			try:
				msg = rospy.wait_for_message(self.image_topic, Image, timeout=1.0)
			except rospy.ROSException:
				continue

			bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
