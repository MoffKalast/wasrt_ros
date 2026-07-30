#pragma once
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/Image.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <stdexcept>
#include <string>

// This package deliberately does not link cv_bridge. ROS's cv_bridge is built against the distro's
// OpenCV 4.5, while everything here builds against JetPack's 4.8; OpenCV does not version its mangled
// symbols, so with both in one process every cv:: call binds to whichever copy the loader reached
// first and the ABI mismatch segfaults on the first call. Wrapping sensor_msgs/Image by hand keeps
// exactly one OpenCV in the process.

namespace wasrt {

inline void assertOpenCVRuntime() {
	const std::string runtime = cv::getVersionString();
	if (runtime != CV_VERSION) {
		ROS_FATAL("OpenCV ABI mismatch: compiled against %s, loaded %s. Something on the link line pulls in a second OpenCV (cv_bridge, image_geometry, opencv_apps); check `ldd` on this binary.", CV_VERSION, runtime.c_str());
		throw std::runtime_error("OpenCV ABI mismatch");
	}
	ROS_INFO("OpenCV %s, header and runtime agree", CV_VERSION);
}

inline int cvTypeForEncoding(const std::string& encoding) {
	if (encoding == "mono8") return CV_8UC1;
	if (encoding == "bgr8" || encoding == "rgb8") return CV_8UC3;
	if (encoding == "bgra8" || encoding == "rgba8") return CV_8UC4;
	return -1;
}

// Zero-copy view over the message buffer; valid only while the message shared_ptr is held.
inline cv::Mat matFromImageMsg(const sensor_msgs::Image& msg) {
	const int type = cvTypeForEncoding(msg.encoding);
	if (type < 0) throw std::runtime_error("unsupported image encoding: " + msg.encoding);

	const size_t row_bytes = static_cast<size_t>(msg.width) * CV_ELEM_SIZE(type);
	if (msg.step < row_bytes) throw std::runtime_error("image step smaller than one row");
	if (msg.data.size() < static_cast<size_t>(msg.step) * msg.height) throw std::runtime_error("image data shorter than step * height");

	return cv::Mat(static_cast<int>(msg.height), static_cast<int>(msg.width), type, const_cast<uint8_t*>(msg.data.data()), static_cast<size_t>(msg.step));
}

inline cv::Mat mono8FromImageMsg(const sensor_msgs::Image& msg) {
	if (msg.encoding != "mono8") throw std::runtime_error("expected mono8, got " + msg.encoding);
	return matFromImageMsg(msg);
}

// Shares the buffer when the message is already bgr8, otherwise converts into scratch.
inline cv::Mat bgrFromImageMsg(const sensor_msgs::Image& msg, cv::Mat& scratch) {
	const cv::Mat view = matFromImageMsg(msg);
	if (msg.encoding == "bgr8") return view;

	if (msg.encoding == "rgb8") cv::cvtColor(view, scratch, cv::COLOR_RGB2BGR);
	else if (msg.encoding == "mono8") cv::cvtColor(view, scratch, cv::COLOR_GRAY2BGR);
	else if (msg.encoding == "bgra8") cv::cvtColor(view, scratch, cv::COLOR_BGRA2BGR);
	else if (msg.encoding == "rgba8") cv::cvtColor(view, scratch, cv::COLOR_RGBA2BGR);
	else throw std::runtime_error("cannot convert " + msg.encoding + " to bgr8");

	return scratch;
}

inline sensor_msgs::ImagePtr imageMsgFromMat(const std_msgs::Header& header, const std::string& encoding, const cv::Mat& mat) {
	const int type = cvTypeForEncoding(encoding);
	if (type < 0) throw std::runtime_error("unsupported image encoding: " + encoding);
	if (mat.type() != type) throw std::runtime_error("mat type does not match encoding " + encoding);

	sensor_msgs::ImagePtr msg(new sensor_msgs::Image());
	msg->header = header;
	msg->height = static_cast<uint32_t>(mat.rows);
	msg->width = static_cast<uint32_t>(mat.cols);
	msg->encoding = encoding;
	msg->is_bigendian = 0;

	const size_t row_bytes = static_cast<size_t>(mat.cols) * mat.elemSize();
	msg->step = static_cast<uint32_t>(row_bytes);
	msg->data.resize(row_bytes * static_cast<size_t>(mat.rows));

	// a cropped Mat is a view with step > row_bytes, so this cannot be a single block copy
	for (int r = 0; r < mat.rows; ++r) std::memcpy(msg->data.data() + row_bytes * static_cast<size_t>(r), mat.ptr(r), row_bytes);

	return msg;
}

}
