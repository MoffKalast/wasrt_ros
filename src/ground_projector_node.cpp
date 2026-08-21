#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <opencv2/imgproc.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include "wasrt_ros/image_msg.h"

class GroundProjector {
public:
	GroundProjector(ros::NodeHandle& nh, ros::NodeHandle& pnh) : tf_listener_(tf_buffer_) {
		pnh.param<std::string>("target_frame", target_frame_, "local");
		pnh.param("z_height", z_height_, 0.0);
		pnh.param("max_range_obstacles", max_range_obstacles_, 3.0);
		pnh.param("max_range_free", max_range_free_, 10.0);
		pnh.param("h_stride_frac", h_stride_frac_, 0.02);
		pnh.param("group_cell_size", group_cell_size_, 0.15);
		pnh.param("obstacle_class", obstacle_class_, 0);
		pnh.param("free_class", free_class_, 1);
		

		double roll_unc_deg;
		pnh.param("roll_unc_deg", roll_unc_deg, 8.0);
		roll_unc_ = roll_unc_deg * M_PI / 180.0;
		pnh.param("roll_pad_factor", roll_pad_factor_, 6.0);

		pnh.param("erode_radius", erode_radius_, 2);
		if (erode_radius_ > 0) erode_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * erode_radius_ + 1, 2 * erode_radius_ + 1));

		std::string info_topic, seg_topic, obstacle_cloud_topic, free_cloud_topic;
		pnh.param<std::string>("camera_info_topic", info_topic, "/camera/image_cropped/camera_info");
		pnh.param<std::string>("seg_topic", seg_topic, "/wasr/image_seg");
		pnh.param<std::string>("obstacle_cloud_topic", obstacle_cloud_topic, "/reliable_cloud");
		pnh.param<std::string>("free_cloud_topic", free_cloud_topic, "/free_cloud");

		info_sub_ = nh.subscribe(info_topic, 1, &GroundProjector::infoCb, this);
		seg_sub_ = nh.subscribe(seg_topic, 1, &GroundProjector::segCb, this, ros::TransportHints().tcpNoDelay());
		obstacle_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(obstacle_cloud_topic, 1);
		free_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>(free_cloud_topic, 1);
	}

private:
	
	std::string target_frame_;
	double z_height_, max_range_obstacles_, max_range_free_, h_stride_frac_, group_cell_size_;
	double roll_unc_, roll_pad_factor_;
	int obstacle_class_, free_class_, erode_radius_;
	cv::Mat erode_kernel_;

	sensor_msgs::CameraInfo::ConstPtr camera_info_;

	bool rays_built_ = false;
	Eigen::Matrix<double, 3, Eigen::Dynamic> d_cam_;
	std::vector<int> grid_u_, grid_v_;

	std::vector<int> proj_u_, proj_v_;
	std::vector<float> proj_xyz_, proj_depth_;

	tf2_ros::Buffer tf_buffer_;
	tf2_ros::TransformListener tf_listener_;

	ros::Subscriber info_sub_, seg_sub_;
	ros::Publisher obstacle_cloud_pub_, free_cloud_pub_;

	void infoCb(const sensor_msgs::CameraInfo::ConstPtr& msg) {
		camera_info_ = msg;
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

	bool buildRays() {
		if (rays_built_)
			return true;

		if (!camera_info_)
			return false;

		Eigen::Matrix<double, 3, 4> P;
		for (int i = 0; i < 12; ++i) P(i / 4, i % 4) = camera_info_->P[i];
		double fx = P(0, 0), fy = P(1, 1), cx = P(0, 2), cy = P(1, 2);

		long w = camera_info_->width;
		long h = camera_info_->height;
		long h_stride = std::max(1L, static_cast<long>(std::lround(h_stride_frac_ * w)));

		// roll uncertainty phi shifts a ray's pitch by ~a*phi with a=(u-cx)/fx, worst at the image sides, so drop the outer columns; pad factor is extra margin
		long roll_cut = static_cast<long>(std::ceil(roll_pad_factor_ * fx * std::tan(roll_unc_)));
		roll_cut = std::min(roll_cut, (w - 1) / 2);

		grid_u_.clear();
		grid_v_.clear();

		// meshgrid(us, vs) raveled row-major: outer loop over v, inner over u
		for (long v = 0; v < h; ++v)
			for (long u = roll_cut; u < w - roll_cut; u += h_stride) {
				grid_u_.push_back(static_cast<int>(u));
				grid_v_.push_back(static_cast<int>(v));
			}

		const int N = static_cast<int>(grid_u_.size());

		// camera optical frame: x right, y down, z forward
		d_cam_.resize(3, N);
		for (int i = 0; i < N; ++i) {
			d_cam_(0, i) = (grid_u_[i] - cx) / fx;
			d_cam_(1, i) = (grid_v_[i] - cy) / fy;
			d_cam_(2, i) = 1.0;
		}

		rays_built_ = true;
		return true;
	}

	bool project(const ros::Time& stamp) {
		if (!buildRays()){
			return false;
		}

		geometry_msgs::TransformStamped tf_msg;
		try {
			tf_msg = tf_buffer_.lookupTransform(target_frame_, camera_info_->header.frame_id, stamp, ros::Duration(0.05));
		} catch (const tf2::TransformException& e) {
			ROS_WARN_THROTTLE(2.0, "tf %s -> %s failed: %s", camera_info_->header.frame_id.c_str(), target_frame_.c_str(), e.what());
			return false;
		}

		Eigen::Matrix4d M = toMatrix(tf_msg);
		Eigen::Matrix3d R = M.block<3, 3>(0, 0);
		Eigen::Vector3d o = M.block<3, 1>(0, 3);

		Eigen::Matrix<double, 3, Eigen::Dynamic> d = R * d_cam_;
		const int N = static_cast<int>(grid_u_.size());

		proj_u_.clear();
		proj_v_.clear();
		proj_xyz_.clear();
		proj_depth_.clear();

		for (int i = 0; i < N; ++i) {
			double dz = d(2, i);
			if (std::abs(dz) <= 1e-9)
				continue;

			double t = (z_height_ - o(2)) / dz;
			if (t <= 0.0)
				continue;
			
			// d_cam has unit z so t is optical-axis depth; per-class range cuts happen in segCb
			proj_u_.push_back(grid_u_[i]);
			proj_v_.push_back(grid_v_[i]);
			proj_xyz_.push_back(static_cast<float>(o(0) + t * d(0, i)));
			proj_xyz_.push_back(static_cast<float>(o(1) + t * d(1, i)));
			proj_xyz_.push_back(static_cast<float>(o(2) + t * d(2, i)));
			proj_depth_.push_back(static_cast<float>(t));
		}
		return true;
	}

	bool dimsMatch(const cv::Mat& img) {
		if (img.rows == static_cast<int>(camera_info_->height) && img.cols == static_cast<int>(camera_info_->width))
			return true;

		ROS_WARN_THROTTLE(2.0, "seg image %dx%d does not match camera_info %dx%d", img.cols, img.rows, camera_info_->width, camera_info_->height);
		return false;
	}

	cv::Mat classMask(const cv::Mat& labels, int cls) {
		cv::Mat mask;
		cv::compare(labels, cls, mask, cv::CMP_EQ);
		// erode each class in full-res label space so unreliable border pixels are dropped before sampling
		if (!erode_kernel_.empty()){
			cv::erode(mask, mask, erode_kernel_);
		}
		return mask;
	}

	// absolute cell indices in the fixed target frame so a ground patch maps to the same cell across frames; keep the first point per cell
	std::vector<float> groupGround(const std::vector<float>& xyz) {
		if (xyz.empty() || group_cell_size_ <= 0.0){
			return xyz;
		}

		std::unordered_set<uint64_t> seen;
		const size_t count = xyz.size() / 3;
		seen.reserve(count * 2);
		std::vector<float> out;
		out.reserve(xyz.size());
		for (size_t i = 0; i < count; ++i) {
			long ix = static_cast<long>(std::floor(static_cast<double>(xyz[3 * i]) / group_cell_size_));
			long iy = static_cast<long>(std::floor(static_cast<double>(xyz[3 * i + 1]) / group_cell_size_));
			uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(ix))) << 32) | static_cast<uint32_t>(static_cast<int32_t>(iy));
			if (seen.insert(key).second) {
				out.push_back(xyz[3 * i]);
				out.push_back(xyz[3 * i + 1]);
				out.push_back(xyz[3 * i + 2]);
			}
		}
		return out;
	}

	void segCb(const sensor_msgs::Image::ConstPtr& msg) {
		if (!project(msg->header.stamp)){
			return;
		}

		// view into msg, which outlives this callback
		cv::Mat labels;
		try {
			labels = wasrt::mono8FromImageMsg(*msg);
		} catch (const std::exception& e) {
			ROS_WARN_THROTTLE(2.0, "seg image unusable: %s", e.what());
			return;
		}

		if (!dimsMatch(labels)){
			return;
		}

		cv::Mat obstacle_mask = classMask(labels, obstacle_class_);
		cv::Mat free_mask = classMask(labels, free_class_);

		const size_t np = proj_u_.size();
		std::vector<float> obstacle_xyz, free_xyz;
		for (size_t i = 0; i < np; ++i) {
			int u = proj_u_[i], v = proj_v_[i];
			double depth = proj_depth_[i];
			if (obstacle_mask.at<uchar>(v, u) && depth <= max_range_obstacles_) {
				obstacle_xyz.push_back(proj_xyz_[3 * i]);
				obstacle_xyz.push_back(proj_xyz_[3 * i + 1]);
				obstacle_xyz.push_back(proj_xyz_[3 * i + 2]);
			}
			if (free_mask.at<uchar>(v, u) && depth <= max_range_free_) {
				free_xyz.push_back(proj_xyz_[3 * i]);
				free_xyz.push_back(proj_xyz_[3 * i + 1]);
				free_xyz.push_back(proj_xyz_[3 * i + 2]);
			}
		}

		obstacle_cloud_pub_.publish(makeXyzCloud(msg->header.stamp, groupGround(obstacle_xyz)));
		free_cloud_pub_.publish(makeXyzCloud(msg->header.stamp, groupGround(free_xyz)));
	}

	sensor_msgs::PointCloud2 makeXyzCloud(const ros::Time& stamp, const std::vector<float>& xyz) {
		sensor_msgs::PointCloud2 cloud;
		cloud.header.stamp = stamp;
		cloud.header.frame_id = target_frame_;
		cloud.height = 1;
		cloud.width = static_cast<uint32_t>(xyz.size() / 3);
		cloud.fields.resize(3);

		const char* names[3] = {"x", "y", "z"};
		for (int i = 0; i < 3; ++i) {
			cloud.fields[i].name = names[i];
			cloud.fields[i].offset = 4 * i;
			cloud.fields[i].datatype = sensor_msgs::PointField::FLOAT32;
			cloud.fields[i].count = 1;
		}

		cloud.is_bigendian = false;
		cloud.point_step = 12;
		cloud.row_step = cloud.point_step * cloud.width;
		cloud.is_dense = true;
		cloud.data.resize(xyz.size() * sizeof(float));

		if (!xyz.empty()){
			std::memcpy(cloud.data.data(), xyz.data(), cloud.data.size());
		}
		
		return cloud;
	}
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "ground_projector_node");
	ros::NodeHandle nh;
	ros::NodeHandle pnh("~");
	wasrt::assertOpenCVRuntime();
	GroundProjector node(nh, pnh);
	ros::spin();
	return 0;
}
