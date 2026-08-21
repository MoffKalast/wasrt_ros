#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/LaserScan.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <boost/bind/bind.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include "wasrt_ros/image_msg.h"

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::LaserScan> SyncPolicy;

class LidarVerifier {
public:
	LidarVerifier(ros::NodeHandle& nh, ros::NodeHandle& pnh) : tf_listener_(tf_buffer_) {
		pnh.param("point_radius", point_radius_, 1);
		pnh.param("publish_preview", publish_preview_, false);
		pnh.param("static_tf", static_tf_, true);
		pnh.param("obstacle_class", obstacle_class_, 0);

		std::string scan_topic, seg_topic, preview_topic, info_topic, out_topic;
		pnh.param<std::string>("scan_topic", scan_topic, "/scan_filtered");
		pnh.param<std::string>("seg_topic", seg_topic, "/wasr/image_seg");
		pnh.param<std::string>("preview_topic", preview_topic, "/wasr/image_preview/compressed");
		pnh.param<std::string>("camera_info_topic", info_topic, "/camera/image_cropped/camera_info");
		pnh.param<std::string>("scan_out_topic", out_topic, "/scan_verified");

		info_sub_ = nh.subscribe(info_topic, 1, &LidarVerifier::infoCb, this);
		scan_pub_ = nh.advertise<sensor_msgs::LaserScan>(out_topic, 1);

		// the preview is best effort
		if (publish_preview_) {
			preview_sub_ = nh.subscribe(preview_topic, 1, &LidarVerifier::previewCb, this, ros::TransportHints().tcpNoDelay());
			image_pub_ = nh.advertise<sensor_msgs::CompressedImage>("/wasr/scan_preview/compressed", 1);
		}

		seg_sub_.subscribe(nh, seg_topic, 1);
		scan_sub_.subscribe(nh, scan_topic, 1);

		sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(4), seg_sub_, scan_sub_));
		sync_->setMaxIntervalDuration(ros::Duration(0.05));
		sync_->registerCallback(boost::bind(&LidarVerifier::syncCb, this, boost::placeholders::_1, boost::placeholders::_2));
	}

private:

	int point_radius_ = 1;
	int obstacle_class_ = 0;
	bool publish_preview_ = false;
	bool static_tf_ = true;

	bool have_info_ = false;
	std::string info_frame_;
	Eigen::Matrix<double, 3, 4> P_;

	bool have_tf_ = false;
	Eigen::Matrix4d tf_cached_;
	tf2_ros::Buffer tf_buffer_;
	tf2_ros::TransformListener tf_listener_;

	sensor_msgs::CompressedImage::ConstPtr latest_preview_;

	ros::Subscriber info_sub_, preview_sub_;
	ros::Publisher scan_pub_, image_pub_;
	message_filters::Subscriber<sensor_msgs::Image> seg_sub_;
	message_filters::Subscriber<sensor_msgs::LaserScan> scan_sub_;
	boost::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

	void previewCb(const sensor_msgs::CompressedImage::ConstPtr& msg) {
		latest_preview_ = msg;
	}

	void infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg) {
		info_frame_ = msg->header.frame_id;

		for (int i = 0; i < 12; ++i)
			P_(i / 4, i % 4) = msg->P[i];

		have_info_ = true;
	}

	static Eigen::Matrix4d toMatrix(const geometry_msgs::TransformStamped& t) {
		const auto& q = t.transform.rotation;
		const auto& tr = t.transform.translation;
		Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
		Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
		M.block<3, 3>(0, 0) = quat.toRotationMatrix();
		M(0, 3) = tr.x;
		M(1, 3) = tr.y;
		M(2, 3) = tr.z;
		return M;
	}

	bool lookupTf(const sensor_msgs::LaserScan& scan, Eigen::Matrix4d& M) {
		if (static_tf_ && have_tf_) {
			M = tf_cached_;
			return true;
		}

		geometry_msgs::TransformStamped t = tf_buffer_.lookupTransform(info_frame_, scan.header.frame_id, scan.header.stamp, ros::Duration(0.1));
		M = toMatrix(t);

		if (static_tf_) { 
			tf_cached_ = M;
			have_tf_ = true;
		}

		return true;
	}

	void publishVerified(const sensor_msgs::LaserScan& scan, const std::vector<float>& ranges) {
		sensor_msgs::LaserScan out;
		out.header = scan.header;
		out.angle_min = scan.angle_min;
		out.angle_max = scan.angle_max;
		out.angle_increment = scan.angle_increment;
		out.time_increment = scan.time_increment;
		out.scan_time = scan.scan_time;
		out.range_min = scan.range_min;
		out.range_max = scan.range_max;
		out.ranges = ranges;
		// intensities belong to the raw returns and carry no meaning for verified points
		out.intensities.clear();
		scan_pub_.publish(out);
	}

	void forwardPreview() {
		if (publish_preview_ && latest_preview_){
			image_pub_.publish(latest_preview_);
		}
	}

	void syncCb(const sensor_msgs::Image::ConstPtr& seg_msg, const sensor_msgs::LaserScan::ConstPtr& scan) {
		if (!have_info_) {
			ROS_WARN_THROTTLE(5.0, "waiting for camera_info, not publishing verified scan");
			forwardPreview();
			return;
		}

		Eigen::Matrix4d M;
		try {
			lookupTf(*scan, M);
		} catch (const tf2::TransformException& e) {
			ROS_WARN_THROTTLE(2.0, "tf %s -> %s failed, not publishing verified scan: %s", scan->header.frame_id.c_str(), info_frame_.c_str(), e.what());
			forwardPreview();
			return;
		}

		const std::vector<float>& ranges = scan->ranges;
		const int n = static_cast<int>(ranges.size());
		const float nan = std::numeric_limits<float>::quiet_NaN();

		std::vector<int> idx;
		idx.reserve(n);
		for (int i = 0; i < n; ++i) {
			float r = ranges[i];
			if (std::isfinite(r) && r >= scan->range_min && r <= scan->range_max){
				idx.push_back(i);
			}
		}

		if (idx.empty()) {
			publishVerified(*scan, std::vector<float>(n, nan));
			forwardPreview();
			return;
		}

		cv::Mat labels;
		try {
			labels = wasrt::mono8FromImageMsg(*seg_msg);
		} catch (const std::exception& e) {
			ROS_WARN_THROTTLE(2.0, "verification failed, not publishing verified scan: %s", e.what());
			forwardPreview();
			return;
		}
		const int w = labels.cols, h = labels.rows;

		const int m = static_cast<int>(idx.size());
		Eigen::Matrix<double, 4, Eigen::Dynamic> pts(4, m);
		for (int k = 0; k < m; ++k) {
			double angle = scan->angle_min + static_cast<double>(idx[k]) * scan->angle_increment;
			double r = ranges[idx[k]];
			pts(0, k) = r * std::cos(angle);
			pts(1, k) = r * std::sin(angle);
			pts(2, k) = 0.0;
			pts(3, k) = 1.0;
		}

		Eigen::Matrix<double, 3, 4> PM = P_ * M;
		Eigen::Matrix<double, 3, Eigen::Dynamic> uvw = PM * pts;

		std::vector<int> uu(m), vv(m);
		std::vector<char> visible(m), verified(m);
		std::vector<float> new_ranges = ranges;
		for (int k = 0; k < m; ++k) {
			double depth = uvw(2, k);
			bool front = depth > 1e-6;
			double safe = front ? depth : 1.0;

			int u = static_cast<int>(std::nearbyint(uvw(0, k) / safe));
			int v = static_cast<int>(std::nearbyint(uvw(1, k) / safe));

			bool vis = front && u >= 0 && u < w && v >= 0 && v < h;
			bool ok = vis && labels.at<uchar>(v, u) == obstacle_class_;

			uu[k] = u;
			vv[k] = v;
			visible[k] = vis;
			verified[k] = ok;

			// a point is kept only if the camera sees it AND it lands on an obstacle pixel; everything else can't be verified
			if (!ok){
				new_ranges[idx[k]] = nan;
			}
		}

		publishVerified(*scan, new_ranges);

		if (publish_preview_ && latest_preview_){
			publishPreviewImage(latest_preview_, uu, vv, visible, verified);
		}
	}

	void publishPreviewImage(const sensor_msgs::CompressedImage::ConstPtr& preview_msg, const std::vector<int>& uu, const std::vector<int>& vv, const std::vector<char>& visible, const std::vector<char>& verified) {
		try {
			cv::Mat img = cv::imdecode(cv::Mat(preview_msg->data), cv::IMREAD_COLOR);

			if (img.empty())
				return;

			for (size_t k = 0; k < uu.size(); ++k) {
				if (!visible[k])
					continue;

				cv::circle(img, cv::Point(uu[k], vv[k]), point_radius_, verified[k] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), -1);
			}

			std::vector<uchar> buf;
			if (!cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 80}))
				return;

			sensor_msgs::CompressedImage out;
			out.header = preview_msg->header;
			out.format = "jpeg";
			out.data = std::move(buf);
			image_pub_.publish(out);

		} catch (const std::exception& e) {
			ROS_WARN_THROTTLE(2.0, "preview rendering failed: %s", e.what());
		}
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "lidar_verifyer_node");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");
	wasrt::assertOpenCVRuntime();
	LidarVerifier node(nh, pnh);
	ros::spin();
	return 0;
}
