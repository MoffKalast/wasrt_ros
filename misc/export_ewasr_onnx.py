#!/usr/bin/env python3
import time
import argparse
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import onnx
import onnxruntime as ort

# eWaSR is stateless, so unlike WaSR-T there is no memory tensor: one image in, one logits map out.
# The resolution is baked in at trace time; edit this to match what the preprocessor produces.
SIZE = (640, 192)

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
		gx = F.interpolate(gx, size=(lx.size(2), lx.size(3)), mode='nearest')
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
		# F.interpolate instead of the torch node's TF.resize: TF.resize defaults to antialias=True,
		# which traces to aten::_upsample_bilinear2d_aa and has no ONNX opset 17 equivalent. These
		# three calls only ever upsample, where antialias is a no-op, so the graphs are numerically
		# identical (verified to float epsilon) and the two nodes still agree.
		size = (f4.size(2), f4.size(3))
		f1 = F.interpolate(f1, size=size, mode='bilinear', align_corners=False)
		f2 = F.interpolate(f2, size=size, mode='bilinear', align_corners=False)
		f3 = F.interpolate(f3, size=size, mode='bilinear', align_corners=False)

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
	"""Recover the architecture from the weights so the export never has to be told the config.

	Kept identical to scripts/ewasr_node.py: the exported ONNX has to describe the same graph the
	torch node would have built from the same checkpoint, otherwise the two nodes silently disagree.
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
		raise ValueError('checkpoint looks like an IMU variant, which is not supported')

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

def default_spec():
	# Published non-IMU eWaSR ResNet-18: 3 classes, ARM mixers, spatial-attention enricher.
	return {
		'block': BasicBlock,
		'layers': [2, 2, 2, 2],
		'backbone_ch': [512, 256, 128, 64],
		'decoder_ch': [512, 256, 128, 64],
		'ch_sim': 128,
		'num_classes': 3,
		'mixer': [ARMMixer, ARMMixer],
		'enricher': [SpatialAttentionMixer],
	}

def build_model(weights, fp16, device):
	if weights:
		sd = load_state_dict(weights)
		spec = describe_checkpoint(sd)
		m = EWaSR(spec).eval()
		m.load_state_dict(sd)
	else:
		spec = default_spec()
		m = EWaSR(spec).eval()
	m = m.to(device)
	if fp16:
		m = m.half()
	return m, spec

def export(model, path, dtype, device):
	image = torch.zeros(1, 3, SIZE[1], SIZE[0], device=device, dtype=dtype)
	kwargs = dict(input_names=['image'], output_names=['logits'], opset_version=17, do_constant_folding=True)
	# dynamo=False forces the TorchScript exporter; older torch builds don't accept the kwarg at all.
	try:
		torch.onnx.export(model, (image,), path, dynamo=False, **kwargs)
	except TypeError:
		torch.onnx.export(model, (image,), path, **kwargs)
	return image

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--weights', default='')
	ap.add_argument('--out', default='ewasr.onnx')
	# The MetaFormer layer_scale parameters initialise at 1e-5, which is subnormal in fp16, so
	# exporting fp32 and letting the TensorRT step handle the cast is the safer path here.
	ap.add_argument('--fp16', action='store_true')
	ap.add_argument('--bench', type=int, default=0)
	args = ap.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	dtype = torch.float16 if args.fp16 else torch.float32
	if args.fp16 and device.type == 'cpu':
		raise RuntimeError('fp16 export needs CUDA; several ops have no cpu half implementation')

	model, spec = build_model(args.weights, args.fp16, device)
	if not args.weights:
		print('no --weights given, exporting randomly initialised %s' % spec_summary(spec))
	else:
		print('loaded %s' % spec_summary(spec))

	image = export(model, args.out, dtype, device)
	print('exported %s (%dx%d, %s)' % (args.out, SIZE[0], SIZE[1], 'fp16' if args.fp16 else 'fp32'))

	onnx.checker.check_model(args.out)
	print('onnx.checker ok')

	providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
	sess = ort.InferenceSession(args.out, providers=providers)
	print('ORT providers:', sess.get_providers())

	with torch.inference_mode():
		ref_logits = model(image)
	feeds = {'image': image.cpu().numpy()}
	ort_logits = sess.run(['logits'], feeds)[0]
	diff = np.abs(ref_logits.float().cpu().numpy() - ort_logits.astype(np.float32)).max()
	print('logits shape %s' % (tuple(ort_logits.shape),))
	print('max abs logits diff torch vs ORT: %.4g' % diff)

	# Class ids are what the ROS node actually publishes, so a diff that only moves logits slightly
	# but flips argmax matters more than the raw magnitude above.
	ref_labels = ref_logits.float().cpu().numpy().argmax(1)
	ort_labels = ort_logits.astype(np.float32).argmax(1)
	print('argmax mismatch: %.4f%% of pixels' % (100.0 * (ref_labels != ort_labels).mean()))

	if args.bench:
		for _ in range(20):
			sess.run(['logits'], feeds)
		t0 = time.time()
		for _ in range(args.bench):
			sess.run(['logits'], feeds)
		print('ORT mean %.2f ms over %d iters' % (1000.0 * (time.time() - t0) / args.bench, args.bench))

if __name__ == '__main__':
	main()
