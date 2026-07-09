#!/usr/bin/env python3
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet101
import onnx
import onnxruntime as ort

# Fixed by the published WaSR-T ResNet-101 weights: 3 classes (0 obstacle, 1 water, 2 sky), context length 5.
NUM_CLASSES = 3
HIST_LEN = 5
SIZE = (320, 96)

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

	def forward(self, feat, mem_in):
		# mem_in holds the previous hist_len conv_in outputs; append the current frame, aggregate, and slide the window
		feat_in = self.conv_in(feat)
		hist_volume = torch.cat([mem_in, feat_in.unsqueeze(1)], dim=1)
		mem_out = hist_volume[:, 1:]
		agg = self.conv_agg(hist_volume.permute(0, 2, 1, 3, 4)).squeeze(2)
		return torch.cat([agg, hist_volume[:, -1]], 1), mem_out

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

	def forward(self, feats, mem_in):
		x, mem_out = self.tcm(feats['out'], mem_in)
		x = self.arm1(x) + self.arm2(feats['skip2'])
		x = self.ffm(feats['skip1'], x)
		return self.aspp(x), mem_out

class WaSRT(nn.Module):
	def __init__(self, num_classes=NUM_CLASSES, hist_len=HIST_LEN):
		super().__init__()
		self.backbone = IntermediateLayerGetter(resnet101_dilated(), {'layer1': 'skip1', 'layer2': 'skip2', 'layer4': 'out'})
		self.decoder = WaSRTDecoder(num_classes, hist_len)

	def forward(self, image, mem_in):
		return self.decoder(self.backbone(image), mem_in)

	def init_mem(self, batch, height, width, device, dtype):
		return torch.zeros(batch, self.decoder.tcm.hist_len, self.decoder.tcm.conv_in.out_channels, height // 8, width // 8, device=device, dtype=dtype)

def load_weights(path):
	sd = torch.load(path, map_location='cpu')
	if 'model' in sd:
		sd = sd['model']
	# strip the prefix torch.compile() adds, in case the weights came from a compiled model
	return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}

def build_model(weights, fp16, device):
	m = WaSRT().eval()
	if weights:
		m.load_state_dict(load_weights(weights))
	m = m.to(device)
	if fp16:
		m = m.half()
	return m

def export(model, path, dtype, device):
	image = torch.zeros(1, 3, SIZE[1], SIZE[0], device=device, dtype=dtype)
	mem = model.init_mem(1, SIZE[1], SIZE[0], device, dtype)
	kwargs = dict(input_names=['image', 'mem_in'], output_names=['logits', 'mem_out'], opset_version=17, do_constant_folding=True)
	# dynamo=False forces the TorchScript exporter; older torch builds don't accept the kwarg at all.
	try:
		torch.onnx.export(model, (image, mem), path, dynamo=False, **kwargs)
	except TypeError:
		torch.onnx.export(model, (image, mem), path, **kwargs)
	return image, mem

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--weights', default='')
	ap.add_argument('--out', default='wasrt.onnx')
	ap.add_argument('--fp16', action='store_true')
	ap.add_argument('--bench', type=int, default=0)
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.float16 if args.fp16 else torch.float32
	model = build_model(args.weights, args.fp16, device)
	image, mem = export(model, args.out, dtype, device)
	print('exported %s (%dx%d, %s)' % (args.out, SIZE[0], SIZE[1], 'fp16' if args.fp16 else 'fp32'))

	onnx.checker.check_model(args.out)
	print('onnx.checker ok')

	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
	sess = ort.InferenceSession(args.out, providers=providers)
	print('ORT providers:', sess.get_providers())

	with torch.inference_mode():
		ref_logits, _ = model(image, mem)
	feeds = {'image': image.cpu().numpy(), 'mem_in': mem.cpu().numpy()}
	ort_logits, ort_mem = sess.run(['logits', 'mem_out'], feeds)
	diff = np.abs(ref_logits.float().cpu().numpy() - ort_logits.astype(np.float32)).max()
	print('max abs logits diff torch vs ORT: %.4g' % diff)

	if args.bench:
		feeds = {'image': image.cpu().numpy(), 'mem_in': mem.cpu().numpy()}
		for _ in range(20):
			feeds['mem_in'] = sess.run(['logits', 'mem_out'], feeds)[1]
		t0 = time.time()
		for _ in range(args.bench):
			feeds['mem_in'] = sess.run(['logits', 'mem_out'], feeds)[1]
		print('ORT mean %.2f ms over %d iters' % (1000.0 * (time.time() - t0) / args.bench, args.bench))

if __name__ == '__main__':
	main()
