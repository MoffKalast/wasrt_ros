#!/usr/bin/env python3
import os
import time
from collections import OrderedDict, deque

import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge

DEFAULT_SIZE = (640, 192)
SEGMENTATION_COLORS = np.array([[247, 195, 37], [41, 167, 224], [90, 75, 164]], np.uint8)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
		w = self.sigmoid(self.bn1(self.conv1(self.global_pool(x))))
		out = w * inp
		if self.last_arm:
			out = self.global_pool(out) * out
		return out

class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features):
		super().__init__()
		self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
		self.act = nn.ReLU()
		self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
		self.drop = nn.Dropout(0.0)

	def forward(self, x):
		return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class ARMMixer(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.arm = AttentionRefinementModule(in_channels=dim, last_arm=False)

	def forward(self, x):
		return self.arm(x)

class SpatialAttentionMixer(nn.Module):
	def __init__(self, kernel_size=3, **kwargs):
		super().__init__()
		self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_out = torch.mean(x, dim=1, keepdim=True)
		max_out, _ = torch.max(x, dim=1, keepdim=True)
		att = self.conv(torch.cat([avg_out, max_out], dim=1))
		return self.sigmoid(att) * x

class MetaFormerBlock(nn.Module):
	def __init__(self, dim, token_mixer, mlp_ratio=4.0):
		super().__init__()
		self.norm1 = nn.BatchNorm2d(dim)
		self.token_mixer = token_mixer(dim=dim)
		self.norm2 = nn.BatchNorm2d(dim)
		self.mlp = Mlp(dim, int(dim * mlp_ratio))
		self.layer_scale_1 = nn.Parameter(1e-5 * torch.ones(dim))
		self.layer_scale_2 = nn.Parameter(1e-5 * torch.ones(dim))

	def forward(self, x):
		x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x))
		x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))
		return x

class PyramidPoolAgg(nn.Module):
	def __init__(self, stride):
		super().__init__()
		self.stride = stride

	def forward(self, inputs):
		_, _, H, W = inputs[0].shape
		H = (H - 1) // self.stride + 1
		W = (W - 1) // self.stride + 1
		return torch.cat([F.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class SIM(nn.Module):
	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.conv_lc = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_lc = nn.BatchNorm2d(ch_out)
		self.conv_gc = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_gc = nn.BatchNorm2d(ch_out)
		self.sigmoid_gc = nn.Sigmoid()
		self.conv_gc1 = nn.Conv2d(ch_in, ch_out, 1)
		self.bn_gc1 = nn.BatchNorm2d(ch_out)

	def forward(self, lx, gx):
		gx = TF.resize(gx, (lx.size(2), lx.size(3)), InterpolationMode.NEAREST)
		lx = self.bn_lc(self.conv_lc(lx))
		multi = self.sigmoid_gc(self.bn_gc(self.conv_gc(gx)))
		gx = self.bn_gc1(self.conv_gc1(gx))
		return lx * multi + gx

class SegHead(nn.Module):
	def __init__(self, ch, num_classes):
		super().__init__()
		self.conv = nn.Conv2d(ch, ch, 1)
		self.bn = nn.BatchNorm2d(ch)
		self.relu = nn.ReLU6(ch)
		self.conv1 = nn.Conv2d(ch, num_classes, 1)

	def forward(self, x):
		return self.conv1(self.relu(self.bn(self.conv(x))))

class EWaSRDecoder(nn.Module):
	def __init__(self, num_classes, ch, ch_sim, mixer, enricher, in_ch=None):
		super().__init__()
		self.ch = list(ch)
		in_ch = list(self.ch) if in_ch is None else list(in_ch)
		self.project = in_ch != self.ch

		# Bottleneck backbones (ResNet-50+) are 4x wider than BasicBlock ones. The training code
		# projects them down to the BasicBlock widths before anything else touches them, so the
		# projection has to happen here, ahead of pool_feats, and the SIMs must see the projected
		# tensors rather than the raw backbone features.
		if self.project:
			self.convs_project1 = nn.Conv2d(in_ch[0], self.ch[0], 1)
			self.convs_project2 = nn.Conv2d(in_ch[1], self.ch[1], 1)
			self.convs_project3 = nn.Conv2d(in_ch[2], self.ch[2], 1)
			self.convs_project4 = nn.Conv2d(in_ch[3], self.ch[3], 1)

		self.pool_feats = PyramidPoolAgg(2)
		self.metaformers = nn.Sequential(*[MetaFormerBlock(sum(self.ch), m) for m in mixer])
		self.metaformers_skip2 = nn.Sequential(*[MetaFormerBlock(self.ch[2], m) for m in enricher])
		self.sim1 = SIM(self.ch[0], ch_sim)
		self.sim2 = SIM(self.ch[1], ch_sim)
		self.sim3 = SIM(self.ch[2], ch_sim)
		self.sim4 = SIM(self.ch[3], ch_sim)
		self.seg_head = SegHead(ch_sim, num_classes)

	def forward(self, out, aux, skip2, skip1):
		if self.project:
			out = self.convs_project1(out)
			aux = self.convs_project2(aux)
			skip2 = self.convs_project3(skip2)
			skip1 = self.convs_project4(skip1)

		tokens = self.metaformers(self.pool_feats([out, aux, skip2, skip1]))
		skip2 = self.metaformers_skip2(skip2)

		# Each token slice covers the channels contributed by one stage, in [out, aux, skip2, skip1] order.
		f1 = self.sim1(out, tokens[:, :self.ch[0]])
		f2 = self.sim2(aux, tokens[:, self.ch[0]:sum(self.ch[:2])])
		f3 = self.sim3(skip2, tokens[:, sum(self.ch[:2]):sum(self.ch[:3])])
		f4 = self.sim4(skip1, tokens[:, sum(self.ch[:3]):])

		# f4 (skip1) is the highest-resolution branch; bring the coarser ones up to it before summing.
		size = (f4.size(2), f4.size(3))
		f1 = TF.resize(f1, size, InterpolationMode.BILINEAR)
		f2 = TF.resize(f2, size, InterpolationMode.BILINEAR)
		f3 = TF.resize(f3, size, InterpolationMode.BILINEAR)

		return self.seg_head(f1 + f2 + f3 + f4)

class EWaSR(nn.Module):
	def __init__(self, spec):
		super().__init__()
		backbone = ResNet(spec['block'], spec['layers'])
		self.backbone = IntermediateLayerGetter(backbone, {'layer1': 'skip1', 'layer2': 'skip2', 'layer3': 'aux', 'layer4': 'out'})
		self.decoder = EWaSRDecoder(
			num_classes=spec['num_classes'],
			ch=spec['decoder_ch'],
			ch_sim=spec['ch_sim'],
			mixer=spec['mixer'],
			enricher=spec['enricher'],
			in_ch=spec['backbone_ch'])

	def forward(self, image):
		f = self.backbone(image)
		return self.decoder(f['out'], f['aux'], f['skip2'], f['skip1'])

def load_state_dict(path):
	try:
		ckpt = torch.load(path, map_location='cpu', weights_only=True)
	except TypeError:
		# weights_only was added in torch 1.13.
		ckpt = torch.load(path, map_location='cpu')
	except Exception:
		# Lightning .ckpt files carry pickled objects that weights_only rejects.
		ckpt = torch.load(path, map_location='cpu', weights_only=False)

	for key in ('state_dict', 'model'):
		if isinstance(ckpt, dict) and key in ckpt:
			ckpt = ckpt[key]
			break

	sd = {}
	for k, v in ckpt.items():
		# LitModel wraps the network as self.model, and torch.compile prefixes _orig_mod.
		k = k.replace('_orig_mod.', '')
		if k.startswith('model.'):
			k = k[len('model.'):]
		sd[k] = v
	return sd

def describe_checkpoint(sd):
	"""Recover the architecture from the weights so the node never has to be told the config.

	Guessing from a --backbone string is how you end up loading a checkpoint into a mismatched
	decoder, which either raises a confusing shape error or, worse, loads and produces nonsense.
	"""
	def count_blocks(layer):
		idx = set()
		prefix = 'backbone.%s.' % layer
		for k in sd:
			if k.startswith(prefix):
				idx.add(int(k[len(prefix):].split('.')[0]))
		return len(idx)

	layers = [count_blocks('layer%d' % i) for i in (1, 2, 3, 4)]
	if min(layers) == 0:
		raise ValueError('checkpoint has no recognisable ResNet backbone layers')

	# Bottleneck blocks have a third conv, BasicBlock does not.
	block = Bottleneck if 'backbone.layer1.0.conv3.weight' in sd else BasicBlock
	expansion = block.expansion
	backbone_ch = [512 * expansion, 256 * expansion, 128 * expansion, 64 * expansion]

	if 'decoder.convs_project1.weight' in sd:
		decoder_ch = [sd['decoder.convs_project%d.weight' % i].shape[0] for i in (1, 2, 3, 4)]
	else:
		decoder_ch = list(backbone_ch)

	ch_sim = sd['decoder.sim1.conv_lc.weight'].shape[0]
	num_classes = sd['decoder.seg_head.conv1.weight'].shape[0]

	if sd['decoder.seg_head.conv.weight'].shape[1] != ch_sim:
		raise ValueError('checkpoint looks like an IMU variant, which this node does not implement')

	def read_mixers(prefix):
		mixers = {}
		for k in sd:
			if not k.startswith(prefix):
				continue
			rest = k[len(prefix):]
			i = int(rest.split('.')[0])
			if '.token_mixer.arm.' in rest:
				mixers[i] = ARMMixer
			elif '.token_mixer.conv.' in rest:
				mixers[i] = SpatialAttentionMixer
		return [mixers[i] for i in sorted(mixers)]

	spec = {
		'block': block,
		'layers': layers,
		'backbone_ch': backbone_ch,
		'decoder_ch': decoder_ch,
		'ch_sim': ch_sim,
		'num_classes': num_classes,
		'mixer': read_mixers('decoder.metaformers.'),
		'enricher': read_mixers('decoder.metaformers_skip2.'),
	}
	if not spec['mixer']:
		raise ValueError('checkpoint has no decoder.metaformers blocks')
	return spec

def spec_summary(spec):
	name = {(BasicBlock, (2, 2, 2, 2)): 'resnet18',
	        (BasicBlock, (3, 4, 6, 3)): 'resnet34',
	        (Bottleneck, (3, 4, 6, 3)): 'resnet50',
	        (Bottleneck, (3, 4, 23, 3)): 'resnet101'}.get((spec['block'], tuple(spec['layers'])))
	name = name or '%s%s' % (spec['block'].__name__, spec['layers'])
	mix = ''.join('C' if m is ARMMixer else 'S' for m in spec['mixer'])
	enr = ''.join('C' if m is ARMMixer else 'S' for m in spec['enricher'])
	projected = spec['decoder_ch'] != spec['backbone_ch']
	return '%s backbone_ch=%s decoder_ch=%s%s ch_sim=%d mixer=%s enricher=%s classes=%d' % (
		name, spec['backbone_ch'], spec['decoder_ch'], ' (projected)' if projected else '',
		spec['ch_sim'], mix, enr, spec['num_classes'])

class EWasrNode:
	def __init__(self):
		rospy.init_node('ewasr_node')

		weights = os.path.expanduser(rospy.get_param('~weights'))
		self.image_topic = rospy.get_param('~image_topic', '/camera/image_cropped')
		seg_topic = rospy.get_param('~seg_topic', '/wasr/image_seg')
		preview_topic = rospy.get_param('~preview_topic', '/wasr/image_preview/compressed')
		self.publish_preview = bool(rospy.get_param('~publish_preview', False))
		self.compressed_in = bool(rospy.get_param('~compressed_input', False))
		self.resize_input = bool(rospy.get_param('~resize_input', True))
		precision = str(rospy.get_param('~precision', 'half')).lower()

		self.size = (int(rospy.get_param('~width', DEFAULT_SIZE[0])), int(rospy.get_param('~height', DEFAULT_SIZE[1])))
		if self.size[0] % 32 or self.size[1] % 32:
			raise ValueError('width/height must both be multiples of 32, got %dx%d' % self.size)

		if not torch.cuda.is_available():
			raise RuntimeError('CUDA is not available; check the torch build and that the container sees the GPU')

		# Input size is fixed, so let cuDNN pick the fastest conv algorithm.
		torch.backends.cudnn.benchmark = True
		self.device = torch.device('cuda')

		if precision not in ('half', 'autocast', 'float'):
			raise ValueError("precision must be one of half, autocast, float; got %r" % precision)
		self.precision = precision
		# 'half' casts the whole graph, including BatchNorm and the layer_scale parameters, whose init
		# value of 1e-5 is subnormal in fp16. 'autocast' keeps those in fp32 and only casts the convs,
		# which is the safer default if the output ever looks noisier on-vehicle than it did in
		# validation. Worth A/B-ing once against 'float' on the same bag.
		self.dtype = torch.float16 if precision == 'half' else torch.float32

		state_dict = load_state_dict(weights)
		spec = describe_checkpoint(state_dict)
		model = EWaSR(spec).eval()
		model.load_state_dict(state_dict)
		self.model = model.to(self.device).to(self.dtype)
		self.spec = spec

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

		rospy.loginfo('eWaSR ready: %s', spec_summary(spec))
		rospy.loginfo('input %dx%d precision=%s topic=%s', self.size[0], self.size[1], self.precision, self.image_topic)

	def _warmup(self):
		# Run cuDNN autotuning and lazy CUDA init now so the first real frame is not stalled for seconds.
		dummy = torch.zeros(1, 3, self.size[1], self.size[0], device=self.device, dtype=self.dtype)
		for _ in range(3):
			self._forward(dummy)
		torch.cuda.synchronize()

	def _forward(self, t):
		with torch.inference_mode():
			if self.precision == 'autocast':
				with torch.cuda.amp.autocast(dtype=torch.float16):
					return self.model(t)
			return self.model(t)

	def infer(self, bgr):
		rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
		t = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0).contiguous()
		t = t.to(self.dtype).div_(255.0)
		t = (t - self.mean) / self.std

		logits = self._forward(t)
		# The decoder emits at 1/4 input resolution. Upsample the logits, not the labels, and argmax
		# afterwards: argmax on the coarse map then nearest-upsampling loses thin obstacles outright.
		logits = F.interpolate(logits.float(), size=(self.size[1], self.size[0]), mode='bilinear', align_corners=False)
		return logits.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()

	def prepare(self, msg):
		if self.compressed_in:
			bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
		else:
			bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

		if bgr is None:
			return None
		if bgr.shape[1] != self.size[0] or bgr.shape[0] != self.size[1]:
			if not self.resize_input:
				rospy.logwarn_throttle(5.0, 'input %dx%d != expected %dx%d, skipping',
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
		label = 'eWaSR %s' % spec_summary(self.spec).split()[0]
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
	EWasrNode().run()

if __name__ == '__main__':
	main()