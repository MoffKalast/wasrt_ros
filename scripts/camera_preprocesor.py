#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from std_msgs.msg import Bool

class Geometry(object):
	def __init__(self, sx, sy, x_offset, y_offset, out_w, out_h): self.sx, self.sy, self.x_offset, self.y_offset, self.out_w, self.out_h = sx, sy, x_offset, y_offset, out_w, out_h

def compute_geometry(orig_w, orig_h, scale, top_pct, bottom_pct, divisor):
	scaled_w = max(divisor, int(round(orig_w * scale)))
	scaled_h = max(divisor, int(round(orig_h * scale)))
	sx = scaled_w / float(orig_w)
	sy = scaled_h / float(orig_h)
	top_cut = int(round(scaled_h * top_pct))
	bottom_cut = int(round(scaled_h * bottom_pct))
	h_after = scaled_h - top_cut - bottom_cut

	if h_after < divisor:
		rospy.logwarn_throttle(5.0, "vertical crop leaves height %d < divisor %d" % (h_after, divisor))

	# divisibility slack removed symmetrically in both axes
	extra_w = scaled_w % divisor
	left_extra = extra_w // 2
	out_w = scaled_w - extra_w
	extra_h = h_after % divisor
	top_extra = extra_h // 2
	out_h = h_after - extra_h
	return Geometry(sx, sy, left_extra, top_cut + top_extra, out_w, out_h)

def transform_camera_info(info, g):
	out = CameraInfo()
	out.header = info.header
	out.height = g.out_h
	out.width = g.out_w
	out.distortion_model = info.distortion_model
	out.D = list(info.D)
	out.R = list(info.R)

	# rectified stream: R stays identity, D stays as-is (zeros); only K and P change
	K = list(info.K)
	K[0] = info.K[0] * g.sx
	K[2] = info.K[2] * g.sx - g.x_offset
	K[4] = info.K[4] * g.sy
	K[5] = info.K[5] * g.sy - g.y_offset
	out.K = K

	P = list(info.P)
	P[0] = info.P[0] * g.sx
	P[2] = info.P[2] * g.sx - g.x_offset
	P[3] = info.P[3] * g.sx
	P[5] = info.P[5] * g.sy
	P[6] = info.P[6] * g.sy - g.y_offset

	out.P = P
	out.binning_x = 1
	out.binning_y = 1
	out.roi.x_offset = 0
	out.roi.y_offset = 0
	out.roi.width = g.out_w
	out.roi.height = g.out_h
	out.roi.do_rectify = info.roi.do_rectify

	return out

class DnnPreprocessNode(object):
	def __init__(self):
		self.scale = rospy.get_param("~downsample", 0.25)
		self.top_pct = rospy.get_param("~crop_top", 0.1)
		self.bottom_pct = rospy.get_param("~crop_bottom", 0.32)
		self.divisor = int(rospy.get_param("~divisor", 32))
		self.publish_preview = bool(rospy.get_param("~publish_preview", False))
		
		camera_topic = rospy.get_param("~camera_topic", "/camera/image_rect/compressed")
		output_topic = rospy.get_param("~output_topic", "/camera/image_cropped")
		enable_topic = rospy.get_param("~enable_topic", "/wasrt/enable")

		if self.top_pct + self.bottom_pct >= 1.0:
			rospy.logfatal("crop_top + crop_bottom must be < 1.0")
			raise rospy.ROSInitException("invalid crop parameters")

		self.bridge = CvBridge()

		self.compressed_input = camera_topic.endswith("compressed")
		# camera_info lives next to the image topic, i.e. without the /compressed suffix
		base_topic = camera_topic[:-len("/compressed")] if self.compressed_input else camera_topic

		self.image_pub = rospy.Publisher(output_topic, Image, queue_size=1)
		self.info_pub = rospy.Publisher(output_topic + "/camera_info", CameraInfo, queue_size=1)
		self.preview_pub = rospy.Publisher(output_topic + "/compressed", CompressedImage, queue_size=1) if self.publish_preview else None

		if self.compressed_input:
			self.image_sub = rospy.Subscriber(camera_topic, CompressedImage, self.compressed_image_cb, queue_size=1, buff_size=2 ** 24)
		else:
			self.image_sub = rospy.Subscriber(camera_topic, Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
		self.info_sub = rospy.Subscriber(base_topic + "/camera_info", CameraInfo, self.info_cb, queue_size=1)

		# power gate for the whole segmentation pipeline: no cropped images -> inference nodes idle.
		# enabled by default so the models get warmed up before the first enable/disable command arrives
		self.enabled = True
		self.enable_sub = rospy.Subscriber(enable_topic, Bool, self.enable_cb, queue_size=1)

		rospy.loginfo("preprocessing %s (%s) -> %s", camera_topic, "CompressedImage" if self.compressed_input else "Image", output_topic)

	def enable_cb(self, msg):
		if msg.data != self.enabled:
			rospy.loginfo("segmentation pipeline %s", "enabled" if msg.data else "disabled")
		self.enabled = msg.data

	def compressed_image_cb(self, msg):
		# gate before decoding so a disabled pipeline costs (almost) nothing
		if not self.enabled:
			return
		self.process(self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8"), msg.header)

	def image_cb(self, msg):
		if not self.enabled:
			return
		self.process(self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), msg.header)

	def process(self, img, header):
		h, w = img.shape[:2]
		g = compute_geometry(w, h, self.scale, self.top_pct, self.bottom_pct, self.divisor)

		if g.out_w <= 0 or g.out_h <= 0:
			rospy.logwarn_throttle(5.0, "degenerate output size, dropping frame")
			return

		resized = cv2.resize(img, (int(round(w * g.sx)), int(round(h * g.sy))), interpolation=cv2.INTER_AREA)
		cropped = resized[g.y_offset:g.y_offset + g.out_h, g.x_offset:g.x_offset + g.out_w]

		out_msg = self.bridge.cv2_to_imgmsg(cropped, encoding="bgr8")
		out_msg.header = header
		self.image_pub.publish(out_msg)

		if self.preview_pub is not None:
			preview_msg = self.bridge.cv2_to_compressed_imgmsg(cropped, dst_format="jpg")
			preview_msg.header = header
			self.preview_pub.publish(preview_msg)

	def info_cb(self, msg):
		g = compute_geometry(msg.width, msg.height, self.scale, self.top_pct, self.bottom_pct, self.divisor)
		self.info_pub.publish(transform_camera_info(msg, g))

if __name__ == "__main__":
	rospy.init_node("dnn_preprocess")
	DnnPreprocessNode()
	rospy.spin()
