#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Bool.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>
#include <stdexcept>
#include "wasrt_ros/geometry.h"
#include "wasrt_ros/image_msg.h"

using namespace wasrt;

class DnnPreprocessNode {
public:
	DnnPreprocessNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
		pnh.param("downsample", scale_, 0.25);
		pnh.param("crop_top", top_pct_, 0.1);
		pnh.param("crop_bottom", bottom_pct_, 0.32);
		pnh.param("divisor", divisor_, 32);
		pnh.param("publish_preview", publish_preview_, false);
		pnh.param("start_enabled", enabled_, true);

		std::string camera_topic, output_topic, enable_topic;
		pnh.param<std::string>("camera_topic", camera_topic, "/camera/image_rect/compressed");
		pnh.param<std::string>("output_topic", output_topic, "/camera/image_cropped");
		pnh.param<std::string>("enable_topic", enable_topic, "/wasr/enable");

		if (top_pct_ + bottom_pct_ >= 1.0) {
			ROS_FATAL("crop_top + crop_bottom must be < 1.0");
			throw std::runtime_error("invalid crop parameters");
		}

		const std::string suffix = "/compressed";
		compressed_input_ = camera_topic.size() >= suffix.size() && camera_topic.compare(camera_topic.size() - suffix.size(), suffix.size(), suffix) == 0;
		std::string base_topic = compressed_input_ ? camera_topic.substr(0, camera_topic.size() - suffix.size()) : camera_topic;
		camera_topic_ = camera_topic;
		nh_ = nh;

		image_pub_ = nh.advertise<sensor_msgs::Image>(output_topic, 1);
		info_pub_ = nh.advertise<sensor_msgs::CameraInfo>(output_topic + "/camera_info", 1);
		
		if (publish_preview_){
			preview_pub_ = nh.advertise<sensor_msgs::CompressedImage>(output_topic + "/compressed", 1);
		}

		if (enabled_){
			subscribeImage();
		}

		// power gate: no cropped images -> inference nodes idle. Enabled by default so models warm up before the first command.
		enable_sub_ = nh.subscribe(enable_topic, 1, &DnnPreprocessNode::enableCb, this);
		info_sub_ = nh.subscribe(base_topic + "/camera_info", 1, &DnnPreprocessNode::infoCb, this);

		ROS_INFO("preprocessing %s (%s) -> %s [%s]", camera_topic.c_str(), compressed_input_ ? "CompressedImage" : "Image", output_topic.c_str(), enabled_ ? "enabled" : "disabled");
	}

private:

	double scale_, top_pct_, bottom_pct_;
	int divisor_;
	bool publish_preview_ = false;
	bool compressed_input_ = false;
	bool enabled_ = true;
	cv::Mat bgr_scratch_;

	ros::NodeHandle nh_;
	std::string camera_topic_;

	ros::Publisher image_pub_, info_pub_, preview_pub_;
	ros::Subscriber image_sub_, info_sub_, enable_sub_;

	void subscribeImage() {
		if (compressed_input_){
			image_sub_ = nh_.subscribe(camera_topic_, 1, &DnnPreprocessNode::compressedImageCb, this, ros::TransportHints().tcpNoDelay());
		}else{
			image_sub_ = nh_.subscribe(camera_topic_, 1, &DnnPreprocessNode::imageCb, this, ros::TransportHints().tcpNoDelay());
		}
	}

	void enableCb(const std_msgs::Bool::ConstPtr& msg) {
		if (msg->data == enabled_){
			return;
		}

		enabled_ = msg->data;
		ROS_INFO("segmentation pipeline %s", enabled_ ? "enabled" : "disabled");

		if (enabled_){
			subscribeImage();
		}else{
			image_sub_.shutdown();
		}
	}

	void compressedImageCb(const sensor_msgs::CompressedImage::ConstPtr& msg) {
		if (!enabled_){
			return;
		}

		cv::Mat img = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
		if (img.empty()) {
			ROS_WARN_THROTTLE(5.0, "failed to decode compressed image");
			return;
		}
		process(img, msg->header);
	}

	void imageCb(const sensor_msgs::Image::ConstPtr& msg) {
		if (!enabled_){
			return;
		}
		try {
			// zero-copy when the publisher already sends bgr8, which mjpeg_usb_cam_node does
			process(wasrt::bgrFromImageMsg(*msg, bgr_scratch_), msg->header);
		} catch (const std::exception& e) {
			ROS_WARN_THROTTLE(5.0, "image unusable: %s", e.what());
		}
	}

	void process(const cv::Mat& img, const std_msgs::Header& header) {
		long w = img.cols;
		long h = img.rows;
		Geometry g = compute_geometry(w, h, scale_, top_pct_, bottom_pct_, divisor_);

		if (g.h_after < divisor_){
			ROS_WARN_THROTTLE(5.0, "vertical crop leaves height %ld < divisor %d", g.h_after, divisor_);
		}
		
		if (g.out_w <= 0 || g.out_h <= 0) {
			ROS_WARN_THROTTLE(5.0, "degenerate output size, dropping frame");
			return;
		}

		cv::Mat resized;
		cv::resize(img, resized, cv::Size(round_half_even(w * g.sx), round_half_even(h * g.sy)), 0, 0, cv::INTER_AREA);
		cv::Mat cropped = resized(cv::Rect(g.x_offset, g.y_offset, g.out_w, g.out_h));

		sensor_msgs::ImagePtr out_msg = wasrt::imageMsgFromMat(header, "bgr8", cropped);
		image_pub_.publish(out_msg);

		if (publish_preview_) {
			std::vector<uchar> buf;
			cv::imencode(".jpg", cropped, buf);
			sensor_msgs::CompressedImage preview_msg;
			preview_msg.header = header;
			preview_msg.format = "jpeg";
			preview_msg.data = std::move(buf);
			preview_pub_.publish(preview_msg);
		}
	}

	void infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg) {
		Geometry g = compute_geometry(msg->width, msg->height, scale_, top_pct_, bottom_pct_, divisor_);

		sensor_msgs::CameraInfo out;
		out.header = msg->header;
		out.height = g.out_h;
		out.width = g.out_w;
		out.distortion_model = msg->distortion_model;
		out.D = msg->D;
		out.R = msg->R;
		scale_intrinsics(msg->K.data(), msg->P.data(), g, out.K.data(), out.P.data());
		out.binning_x = 1;
		out.binning_y = 1;
		out.roi.x_offset = 0;
		out.roi.y_offset = 0;
		out.roi.width = g.out_w;
		out.roi.height = g.out_h;
		out.roi.do_rectify = msg->roi.do_rectify;
		info_pub_.publish(out);
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "dnn_preprocess");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");
	wasrt::assertOpenCVRuntime();
	DnnPreprocessNode node(nh, pnh);
	ros::spin();
	return 0;
}