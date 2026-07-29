#!/usr/bin/env python3
import time
from collections import OrderedDict
import os

import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

# Fixed by the published WaSR-T ResNet-101 weights: 3 classes (0 obstacle, 1 water, 2 sky), context length 5.
NUM_CLASSES = 3
HIST_LEN = 5
DEFAULT_SIZE = (640, 192)
SEGMENTATION_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def resnet101_dilated():
	try:
		return resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
	except TypeError:
		return resnet101(pretrained=False, replace_stride_with_dilation=[False, True, True])

class IntermediateLayerGetter(nn.ModuleDict):
	def __init__(self, model, return_layers):
		layers = OrderedDict()
		remaining = set(return_layers)
		for name, module in model.named_children():
			layers[name] = module
			remaining.discard(name)
			if not remaining:
				break
		super().__init__(layers)
		self.return_layers = dict(return_layers)

	def forward(self, x):
		out = OrderedDict()
		for name, module in self.items():
			x = module(x)
			if name in self.return_layers:
				out[self.return_layers[name]] = x
		return out

class TemporalContextModule(nn.Module):
	def __init__(self, in_features, hist_len):
		super().__init__()
		self.conv_in = nn.Conv2d(in_features, in_features // 2, 1)
		self.conv_agg = nn.Conv3d(in_features // 2, in_features // 2, (hist_len + 1, 3, 3), padding=(0, 1, 1))
		self.hist_len = hist_len
		self._mem = None

	def clear_state(self):
		self._mem = None

	def forward(self, feat):
		feat_in = self.conv_in(feat)
		# On the first frame the history buffer is seeded with copies of that frame.
		if self._mem is None:
			self._mem = feat_in.unsqueeze(1).repeat(1, self.hist_len, 1, 1, 1)
		hist_volume = torch.cat([self._mem, feat_in.unsqueeze(1)], dim=1)
		self._mem = hist_volume[:, 1:]
		agg = self.conv_agg(hist_volume.permute(0, 2, 1, 3, 4)).squeeze(2)
		return torch.cat([agg, hist_volume[:, -1]], 1)

class AttentionRefinementModule(nn.Module):
	def __init__(self, in_channels, last_arm=False):
		super().__init__()
		self.last_arm = last_arm
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.conv1 = nn.Conv2d(in_channels, in_channels, 1)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		inp = x
		x = self.bn1(self.conv1(self.global_pool(x)))
		out = self.sigmoid(x) * inp
		if self.last_arm:
			out = self.global_pool(out) * out
		return out

class FeatureFusionModule(nn.Module):
	def __init__(self, bg_channels, sm_channels, num_features):
		super().__init__()
		self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
		self.conv1 = nn.Conv2d(bg_channels + sm_channels, num_features, 3, padding=1)
		self.bn1 = nn.BatchNorm2d(num_features)
		self.relu = nn.ReLU(inplace=True)
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.conv2 = nn.Conv2d(num_features, num_features, 1)
		self.conv3 = nn.Conv2d(num_features, num_features, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x_big, x_small):
		if x_big.size(2) > x_small.size(2):
			x_small = self.upsampling(x_small)
		x = self.relu(self.bn1(self.conv1(torch.cat((x_big, x_small), 1))))
		weights = self.sigmoid(self.conv3(self.conv2(self.global_pool(x))))
		return x + weights * x

class ASPPv2Conv(nn.Sequential):
	def __init__(self, in_channels, out_channels, dilation):
		super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=True))

class ASPPv2(nn.Module):
	def __init__(self, in_channels, atrous_rates, out_channels):
		super().__init__()
		self.convs = nn.ModuleList([ASPPv2Conv(in_channels, out_channels, r) for r in atrous_rates])

	def forward(self, x):
		return torch.stack([conv(x) for conv in self.convs]).sum(0)

class WaSRTDecoder(nn.Module):
	def __init__(self, num_classes, hist_len):
		super().__init__()
		self.arm1 = AttentionRefinementModule(2048)
		self.arm2 = nn.Sequential(AttentionRefinementModule(512, last_arm=True), nn.Conv2d(512, 2048, 1))
		self.tcm = TemporalContextModule(2048, hist_len)
		self.ffm = FeatureFusionModule(256, 2048, 1024)
		self.aspp = ASPPv2(1024, [6, 12, 18, 24], num_classes)

	def forward(self, feats):
		x = self.tcm(feats['out'])
		x = self.arm1(x) + self.arm2(feats['skip2'])
		x = self.ffm(feats['skip1'], x)
		return self.aspp(x)

	def clear_state(self):
		self.tcm.clear_state()

class WaSRT(nn.Module):
	def __init__(self, num_classes=NUM_CLASSES, hist_len=HIST_LEN):
		super().__init__()
		self.backbone = IntermediateLayerGetter(resnet101_dilated(), {'layer1': 'skip1', 'layer2': 'skip2', 'layer4': 'out'})
		self.decoder = WaSRTDecoder(num_classes, hist_len)

	def forward(self, image):
		return self.decoder(self.backbone(image))

	def clear_state(self):
		self.decoder.clear_state()

def load_weights(path):
	sd = torch.load(path, map_location='cpu')
	if 'model' in sd:
		sd = sd['model']
	# Strip the prefix torch.compile() adds, in case the weights came from a compiled model.
	return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

class WasrTNode:
	def __init__(self):
		rospy.init_node('wasr_t_node')
		weights = rospy.get_param('~weights')
		self.image_topic = rospy.get_param('~image_topic', '/camera/image_cropped')
		seg_topic = rospy.get_param('~seg_topic', '/wasr/image_seg')
		preview_topic = rospy.get_param('~preview_topic', '/wasr/image_preview/compressed')
		self.publish_preview = bool(rospy.get_param('~publish_preview', False))

		self.size = (int(rospy.get_param('~width', DEFAULT_SIZE[0])), int(rospy.get_param('~height', DEFAULT_SIZE[1])))
		if self.size[0] % 32 or self.size[1] % 32:
			rospy.logfatal('width/height must both be multiples of 32, got %dx%d', self.size[0], self.size[1])
			raise ValueError('invalid input size')

		# Input size is fixed, so let cuDNN pick the fastest conv algorithm (notably for the dilated layers).
		torch.backends.cudnn.benchmark = True
		self.device = torch.device('cuda')
		self.dtype = torch.float16

		model = WaSRT().eval()
		model.load_state_dict(load_weights(os.path.expanduser(weights)))
		self.model = model.to(self.device).half()
		self.model.clear_state()

		# Normalization runs on the GPU so we only upload uint8 (a quarter of the float32 transfer).
		self.mean = torch.tensor(IMAGENET_MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
		self.std = torch.tensor(IMAGENET_STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

		self.bridge = CvBridge()
		self.pub_seg = rospy.Publisher(seg_topic, Image, queue_size=1)
		self.pub_preview = rospy.Publisher(preview_topic, CompressedImage, queue_size=1) if self.publish_preview else None

		self._warmup()
		rospy.loginfo('WaSR-T ready: size=%dx%d', self.size[0], self.size[1])

	def _warmup(self):
		# Run cuDNN autotuning / lazy CUDA init now so the first real frame is not stalled for seconds.
		dummy = torch.zeros(1, 3, self.size[1], self.size[0], device=self.device, dtype=self.dtype)
		with torch.inference_mode():
			for _ in range(3):
				self.model(dummy)
		torch.cuda.synchronize()
		self.model.clear_state()

	def infer(self, bgr):
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).to(self.dtype).div_(255.0)
		t = (t - self.mean) / self.std
		with torch.inference_mode():
			logits = self.model(t)
		# Backbone output stride leaves logits coarser than the input; upsample back to label resolution.
		logits = F.interpolate(logits, size=(self.size[1], self.size[0]), mode='bilinear', align_corners=False)
		return logits.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()

	def publish(self, labels, original_bgr, header):
		seg_msg = self.bridge.cv2_to_imgmsg(labels, encoding='mono8')
		seg_msg.header = header
		self.pub_seg.publish(seg_msg)

		if self.pub_preview is not None:
			color_bgr = cv2.cvtColor(SEGMENTATION_COLORS[labels], cv2.COLOR_RGB2BGR)
			overlay = cv2.addWeighted(original_bgr, 0.5, color_bgr, 0.5, 0.0)
			cv2.putText(overlay, "WaSR-T (MaSTr1478)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.putText(overlay, "WaSR-T (MaSTr1478)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
			ok, jpg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
			if ok:
				msg = CompressedImage()
				msg.header = header
				msg.format = "jpeg"
				msg.data = jpg.tobytes()
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
			if bgr.shape[1] != self.size[0] or bgr.shape[0] != self.size[1]:
				rospy.logwarn_throttle(5.0, 'input %dx%d != expected %dx%d, skipping' % (bgr.shape[1], bgr.shape[0], self.size[0], self.size[1]))
				continue
			t0 = time.time()
			labels = self.infer(bgr)
			rospy.loginfo('infer %.1f ms', 1000.0 * (time.time() - t0))
			self.publish(labels, bgr, msg.header)

def main():
	WasrTNode().run()

if __name__ == '__main__':
	main()
