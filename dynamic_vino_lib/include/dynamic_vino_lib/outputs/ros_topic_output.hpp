// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header file with declaration for RosTopicOutput Class
 * @file ros_topic_output.hpp
 */

#ifndef DYNAMIC_VINO_LIB__OUTPUTS__ROS_TOPIC_OUTPUT_HPP_
#define DYNAMIC_VINO_LIB__OUTPUTS__ROS_TOPIC_OUTPUT_HPP_

#include <object_msgs/msg/object.hpp>
#include <object_msgs/msg/object_in_box.hpp>
#include <object_msgs/msg/objects_in_boxes.hpp>
#include <people_msgs/msg/emotion.hpp>
#include <people_msgs/msg/emotions_stamped.hpp>
#include <people_msgs/msg/age_gender.hpp>
#include <people_msgs/msg/age_gender_stamped.hpp>
#include <people_msgs/msg/head_pose.hpp>
#include <people_msgs/msg/head_pose_stamped.hpp>
#include <people_msgs/msg/object_in_mask.hpp>
#include <people_msgs/msg/objects_in_masks.hpp>
#include <people_msgs/msg/reidentification.hpp>
#include <people_msgs/msg/reidentification_stamped.hpp>
#include <people_msgs/msg/person_attribute.hpp>
#include <people_msgs/msg/person_attribute_stamped.hpp>
#include <people_msgs/msg/landmark.hpp>
#include <people_msgs/msg/landmark_stamped.hpp>
#include <people_msgs/msg/vehicle_attribs.hpp>
#include <people_msgs/msg/vehicle_attribs_stamped.hpp>
#include <people_msgs/msg/license_plate.hpp>
#include <people_msgs/msg/license_plate_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include <memory>
#include <string>
#include <vector>

#include "dynamic_vino_lib/inferences/face_detection.hpp"
#include "dynamic_vino_lib/outputs/base_output.hpp"

namespace Outputs
{
/**
 * @class RosTopicOutput
 * @brief This class handles and publish the detection result with ros topic.
 */
class RosTopicOutput : public BaseOutput
{
public:
  explicit RosTopicOutput(std::string output_name_, 
    const rclcpp::Node::SharedPtr node=nullptr);
  /**
   * @brief Calculate the camera matrix of a frame.
   * @param[in] A frame.
   */
  void feedFrame(const cv::Mat &) override;
  /**
   * @brief Publish all the detected infomations generated by the accept
   * functions with ros topic.
   */
  void handleOutput() override;
  /**
   * @brief Generate ros topic infomation according to
   * the license plate detection result.
   * @param[in] results a bundle of license plate detection results.
   */
  void accept(const std::vector<dynamic_vino_lib::LicensePlateDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the vehicle attributes detection result.
   * @param[in] results a bundle of vehicle attributes detection results.
   */
  void accept(const std::vector<dynamic_vino_lib::VehicleAttribsDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the face reidentification result.
   * @param[in] results a bundle of face reidentification results.
   */
  void accept(const std::vector<dynamic_vino_lib::FaceReidentificationResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the landmarks detection result.
   * @param[in] results a bundle of landmarks detection results.
   */
  void accept(const std::vector<dynamic_vino_lib::LandmarksDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the person attributes detection result.
   * @param[in] results a bundle of person attributes detection results.
   */
  void accept(const std::vector<dynamic_vino_lib::PersonAttribsDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the person reidentification result.
   * @param[in] results a bundle of person reidentification results.
   */
  void accept(const std::vector<dynamic_vino_lib::PersonReidentificationResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the object segmentation result.
   * @param[in] results a bundle of object segmentation results.
   */
  void accept(const std::vector<dynamic_vino_lib::ObjectSegmentationResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the object segmentation result.
   * @param[in] results a bundle of object segmentation maskrcnn results.
   */
  void accept(const std::vector<dynamic_vino_lib::ObjectSegmentationMaskrcnnResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the object detection result.
   * @param[in] results a bundle of object detection results.
   */
  void accept(const std::vector<dynamic_vino_lib::ObjectDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the face detection result.
   * @param[in] An face detection result objetc.
   */
  void accept(const std::vector<dynamic_vino_lib::FaceDetectionResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the emotion detection result.
   * @param[in] An emotion detection result objetc.
   */
  void accept(const std::vector<dynamic_vino_lib::EmotionsResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the age gender detection result.
   * @param[in] An age gender detection result objetc.
   */
  void accept(const std::vector<dynamic_vino_lib::AgeGenderResult> &) override;
  /**
   * @brief Generate ros topic infomation according to
   * the headpose detection result.
   * @param[in] An head pose detection result objetc.
   */
  void accept(const std::vector<dynamic_vino_lib::HeadPoseResult> &) override;

protected:
  const std::string topic_name_;
  std::shared_ptr<rclcpp::Node> node_;
  rclcpp::Publisher<people_msgs::msg::LicensePlateStamped>::SharedPtr pub_license_plate_;
  std::shared_ptr<people_msgs::msg::LicensePlateStamped> license_plate_topic_;
  rclcpp::Publisher<people_msgs::msg::VehicleAttribsStamped>::SharedPtr pub_vehicle_attribs_;
  std::shared_ptr<people_msgs::msg::VehicleAttribsStamped> vehicle_attribs_topic_;
  rclcpp::Publisher<people_msgs::msg::LandmarkStamped>::SharedPtr pub_landmarks_;
  std::shared_ptr<people_msgs::msg::LandmarkStamped> landmarks_topic_;
  rclcpp::Publisher<people_msgs::msg::ReidentificationStamped>::SharedPtr pub_face_reid_;
  std::shared_ptr<people_msgs::msg::ReidentificationStamped> face_reid_topic_;
  rclcpp::Publisher<people_msgs::msg::PersonAttributeStamped>::SharedPtr pub_person_attribs_;
  std::shared_ptr<people_msgs::msg::PersonAttributeStamped> person_attribs_topic_;
  rclcpp::Publisher<people_msgs::msg::ReidentificationStamped>::SharedPtr pub_person_reid_;
  std::shared_ptr<people_msgs::msg::ReidentificationStamped> person_reid_topic_;
  rclcpp::Publisher<people_msgs::msg::ObjectsInMasks>::SharedPtr pub_segmented_object_;
  std::shared_ptr<people_msgs::msg::ObjectsInMasks> segmented_objects_topic_;
  rclcpp::Publisher<object_msgs::msg::ObjectsInBoxes>::SharedPtr pub_detected_object_;
  std::shared_ptr<object_msgs::msg::ObjectsInBoxes> detected_objects_topic_;
  rclcpp::Publisher<object_msgs::msg::ObjectsInBoxes>::SharedPtr pub_face_;
  std::shared_ptr<object_msgs::msg::ObjectsInBoxes> faces_topic_;
  rclcpp::Publisher<people_msgs::msg::EmotionsStamped>::SharedPtr pub_emotion_;
  std::shared_ptr<people_msgs::msg::EmotionsStamped> emotions_topic_;
  rclcpp::Publisher<people_msgs::msg::AgeGenderStamped>::SharedPtr pub_age_gender_;
  std::shared_ptr<people_msgs::msg::AgeGenderStamped> age_gender_topic_;
  rclcpp::Publisher<people_msgs::msg::HeadPoseStamped>::SharedPtr pub_headpose_;
  std::shared_ptr<people_msgs::msg::HeadPoseStamped> headpose_topic_;
};
}  // namespace Outputs
#endif  // DYNAMIC_VINO_LIB__OUTPUTS__ROS_TOPIC_OUTPUT_HPP_
