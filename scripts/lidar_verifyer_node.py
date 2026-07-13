#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import tf2_ros
import message_filters
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from cv_bridge import CvBridge

class LidarVerifyer:
	def __init__(self):
		self.bridge = CvBridge()
		self.camera_info = None
		self.P = None
		self.tf_matrix = None
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		self.point_radius = rospy.get_param("~point_radius", 1)
		self.publish_preview = bool(rospy.get_param("~publish_preview", False))
		# the lidar -> camera transform is normally fixed; cache it after the first lookup to keep the hot path tf-free
		self.static_tf = bool(rospy.get_param("~static_tf", True))
		self.obstacle_class = int(rospy.get_param("~obstacle_class", 0))
		scan_topic = rospy.get_param("~scan_topic", "/scan_filtered")
		seg_topic = rospy.get_param("~seg_topic", "/wasrt/image_seg")
		preview_topic = rospy.get_param("~preview_topic", "/wasrt/image_preview/compressed")
		info_topic = rospy.get_param("~camera_info_topic", "/camera/image_cropped/camera_info")
		out_topic = rospy.get_param("~scan_out_topic", "/scan_verified")

		self.camera_info_sub = rospy.Subscriber(info_topic, CameraInfo, self.info_cb, queue_size=1)
		self.scan_pub = rospy.Publisher(out_topic, LaserScan, queue_size=1)

		# the preview image is deliberately NOT part of the synchronizer: verification must keep
		# running even when no preview images arrive, so the latest one is just cached on the side
		self.latest_preview = None
		if self.publish_preview:
			self.preview_sub = rospy.Subscriber(preview_topic, CompressedImage, self.preview_cb, queue_size=1, buff_size=2 ** 24)
			self.image_pub = rospy.Publisher("/wasrt/scan_preview/compressed", CompressedImage, queue_size=1)

		self.sync = message_filters.ApproximateTimeSynchronizer([message_filters.Subscriber(seg_topic, Image), message_filters.Subscriber(scan_topic, LaserScan)], queue_size=4, slop=0.05)
		self.sync.registerCallback(self.sync_cb)

	def preview_cb(self, msg):
		self.latest_preview = msg

	def info_cb(self, msg):
		self.camera_info = msg
		self.P = np.asarray(msg.P, dtype=np.float64).reshape(3, 4)

	def transform_matrix(self, tf_msg):
		q = tf_msg.transform.rotation
		t = tf_msg.transform.translation
		M = quaternion_matrix([q.x, q.y, q.z, q.w])
		M[0:3, 3] = [t.x, t.y, t.z]
		return M

	def lookup_tf(self, scan):
		if self.static_tf and self.tf_matrix is not None:
			return self.tf_matrix
		tf_msg = self.tf_buffer.lookup_transform(self.camera_info.header.frame_id, scan.header.frame_id, scan.header.stamp, rospy.Duration(0.1))
		M = self.transform_matrix(tf_msg)
		if self.static_tf:
			self.tf_matrix = M
		return M

	def publish_verified(self, scan, ranges):
		out = LaserScan()
		out.header = scan.header
		out.angle_min = scan.angle_min
		out.angle_max = scan.angle_max
		out.angle_increment = scan.angle_increment
		out.time_increment = scan.time_increment
		out.scan_time = scan.scan_time
		out.range_min = scan.range_min
		out.range_max = scan.range_max
		out.ranges = ranges.tolist()
		# intensities are dropped: they belong to the raw returns and carry no meaning for verified points
		out.intensities = []
		self.scan_pub.publish(out)

	def sync_cb(self, seg_msg, scan):
		if self.camera_info is None:
			rospy.logwarn_throttle(5.0, "waiting for camera_info, not publishing verified scan")
			self.forward_preview()
			return
		try:
			M = self.lookup_tf(scan)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logwarn_throttle(2.0, "tf %s -> %s failed, not publishing verified scan: %s" % (scan.header.frame_id, self.camera_info.header.frame_id, e))
			self.forward_preview()
			return

		try:
			ranges = np.asarray(scan.ranges, dtype=np.float64)
			idx = np.flatnonzero(np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max))
			if idx.size == 0:
				# no valid returns to verify: the verified scan is legitimately empty, not a failure
				self.publish_verified(scan, np.full(ranges.shape, np.nan))
				self.forward_preview()
				return

			angles = scan.angle_min + idx * scan.angle_increment
			r = ranges[idx]
			pts = np.stack([r * np.cos(angles), r * np.sin(angles), np.zeros_like(r), np.ones_like(r)], axis=0)

			labels = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding="mono8")
			h, w = labels.shape[:2]

			uvw = self.P @ (M @ pts)
			depths = uvw[2]
			front = depths > 1e-6
			# avoid div-by-zero warnings for behind-camera points; their uv values are discarded by the mask anyway
			safe_depths = np.where(front, depths, 1.0)
			uu = np.rint(uvw[0] / safe_depths).astype(np.int64)
			vv = np.rint(uvw[1] / safe_depths).astype(np.int64)
			visible = front & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)

			# a point is kept only if the camera sees it AND it lands on an obstacle pixel; unseen points can't be verified
			verified = np.zeros(idx.size, dtype=bool)
			verified[visible] = labels[vv[visible], uu[visible]] == self.obstacle_class
			rejected = ~verified

			new_ranges = ranges.copy()
			new_ranges[idx[rejected]] = np.nan
			self.publish_verified(scan, new_ranges)
		except Exception as e:
			rospy.logwarn_throttle(2.0, "verification failed, not publishing verified scan: %s" % e)
			self.forward_preview()
			return

		if self.latest_preview is not None:
			self.publish_preview_image(self.latest_preview, uu, vv, visible, verified)

	def forward_preview(self):
		if self.latest_preview is not None:
			self.image_pub.publish(self.latest_preview)

	def publish_preview_image(self, preview_msg, uu, vv, visible, verified):
		try:
			img = self.bridge.compressed_imgmsg_to_cv2(preview_msg, "bgr8")
			for x, y, ok in zip(uu[visible], vv[visible], verified[visible]):
				cv2.circle(img, (int(x), int(y)), self.point_radius, (0, 255, 0) if ok else (0, 0, 255), -1)
			ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
			if not ok:
				return
			out = CompressedImage()
			out.header = preview_msg.header
			out.format = "jpeg"
			out.data = buf.tobytes()
			self.image_pub.publish(out)
		except Exception as e:
			rospy.logwarn_throttle(2.0, "preview rendering failed: %s" % e)

if __name__ == "__main__":
	rospy.init_node("lidar_verifyer_node")
	LidarVerifyer()
	rospy.spin()
