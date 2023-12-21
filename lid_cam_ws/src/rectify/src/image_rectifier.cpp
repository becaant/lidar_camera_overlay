// taskset -c 0,1,2 ros2 run rectify image_rectifier
#include <chrono>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include "std_msgs/msg/header.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using std::placeholders::_1;

class ImageSubscriber : public rclcpp::Node
{
public:
  ImageSubscriber()
    : Node("image_subscriber")
  {
    cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    

    rclcpp::SubscriptionOptions options;
    options.callback_group = cb_group_;

    subscription_1 = create_subscription<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_1/image", 10, std::bind(&ImageSubscriber::imageCallback1, this, _1), options);
    subscription_2 = create_subscription<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_2/image", 10, std::bind(&ImageSubscriber::imageCallback2, this, _1), options);
    subscription_3 = create_subscription<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_3/image", 10, std::bind(&ImageSubscriber::imageCallback3, this, _1), options);
    subscription_4 = create_subscription<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_4/image", 10, std::bind(&ImageSubscriber::imageCallback4, this, _1), options);
    publisher_1 = this->create_publisher<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_1/image_rect", 10);
    publisher_2 = this->create_publisher<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_2/image_rect", 10);
    publisher_3 = this->create_publisher<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_3/image_rect", 10);
    publisher_4 = this->create_publisher<sensor_msgs::msg::Image>(
      "/lucid_vision/camera_4/image_rect", 10);
  }

private:
  void imageCallback1(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Start the timer
    auto start = std::chrono::steady_clock::now();
    // Convert sensor_msgs::Image to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Process the received image data here
    cv::Mat distorted_img = cv_ptr->image;

    float camera_matrix_data[3][3] = {{1254.926067, 0.000000, 1026.995770},
                                      {0.000000, 1255.840035, 771.593421},
                                      {0.000000, 0.000000, 1.000000}};
    cv::Mat camera_matrix(3, 3, CV_32F, camera_matrix_data);
    std::cout << camera_matrix << std::endl;

    float distortion_coefficients_data[5] = {-0.199586, 0.067045, -0.000109, 0.000466, 0.000000};
    cv::Mat distortion_coefficients(1, 5, CV_32F, distortion_coefficients_data);
    std::cout << distortion_coefficients << std::endl;

    cv::Mat undistorted_img;
    cv::undistort(distorted_img, undistorted_img, camera_matrix, distortion_coefficients);

    // cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::resizeWindow("view", 512, 300);
    // cv::imshow("view", distorted_img);
    // cv::waitKey(30);

    // cv::namedWindow("Undistorted image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Undistorted image", 512, 300);
    // cv::imshow("Undistorted image", undistorted_img);
    // cv::waitKey(30);

    // Example: Print image dimensions
    // RCLCPP_INFO(get_logger(), "Received image 1: width = %d, height = %d", msg->width, msg->height);

    // Publish the undistorted image
    sensor_msgs::msg::Image::SharedPtr image_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", undistorted_img).toImageMsg();
    image_out->header = msg->header;
    publisher_1->publish(*image_out);

    // Stop the timer
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print the execution time
    RCLCPP_INFO(get_logger(), "Image processing took %lld milliseconds", duration);

  }

  void imageCallback2(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Start the timer
    // auto start = std::chrono::steady_clock::now();
    // Convert sensor_msgs::Image to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Process the received image data here
    cv::Mat distorted_img = cv_ptr->image;

    float camera_matrix_data[3][3] = {{1282.497049, 0.000000, 1029.358980}, 
                                       {0.000000, 1283.368918, 751.809412}, 
                                       {0.000000, 0.000000, 1.000000}};
    cv::Mat camera_matrix(3, 3, CV_32F, camera_matrix_data);

    float distortion_coefficients_data[5] = {-0.217783, 0.086682, 0.000142, 0.000268, 0.000000};

    cv::Mat distortion_coefficients(1, 5, CV_32F, distortion_coefficients_data);
    cv::Mat undistorted_img;
    cv::undistort(distorted_img, undistorted_img, camera_matrix, distortion_coefficients);

    // cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::resizeWindow("view", 512, 300);
    // cv::imshow("view", distorted_img);
    // cv::waitKey(30);

    // cv::namedWindow("Undistorted image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Undistorted image", 512, 300);
    // cv::imshow("Undistorted image", undistorted_img);
    // cv::waitKey(30);

    // Example: Print image dimensions
    // RCLCPP_INFO(get_logger(), "Received image 2: width = %d, height = %d", msg->width, msg->height);

    // Publish the undistorted image
    sensor_msgs::msg::Image::SharedPtr image_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", undistorted_img).toImageMsg();
    image_out->header = msg->header;
    publisher_2->publish(*image_out);

    // Stop the timer
    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print the execution time
    // RCLCPP_INFO(get_logger(), "Image processing took %lld milliseconds", duration);

  }

  void imageCallback3(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Start the timer
    // auto start = std::chrono::steady_clock::now();
    // Convert sensor_msgs::Image to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Process the received image data here
    cv::Mat distorted_img = cv_ptr->image;

    float camera_matrix_data[3][3] = {{1286.923162, 0.000000, 1022.168149}, 
                                       {0.000000, 1287.474163, 766.738280}, 
                                       {0.000000, 0.000000, 1.000000}};

    cv::Mat camera_matrix(3, 3, CV_32F, camera_matrix_data);

    float distortion_coefficients_data[5] = {-0.216417, 0.085101, 0.000401, 0.000181, 0.000000};
    cv::Mat distortion_coefficients(1, 5, CV_32F, distortion_coefficients_data);
    cv::Mat undistorted_img;
    cv::undistort(distorted_img, undistorted_img, camera_matrix, distortion_coefficients);

    // cv::namedWindow("view", cv::WINDOW_NORMAL);
    // cv::resizeWindow("view", 512, 300);
    // cv::imshow("view", distorted_img);
    // cv::waitKey(30);

    // cv::namedWindow("Undistorted image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Undistorted image", 512, 300);
    // cv::imshow("Undistorted image", undistorted_img);
    // cv::waitKey(30);

    // Example: Print image dimensions
    // RCLCPP_INFO(get_logger(), "Received image 3 : width = %d, height = %d", msg->width, msg->height);

    // Publish the undistorted image
    sensor_msgs::msg::Image::SharedPtr image_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", undistorted_img).toImageMsg();
    image_out->header = msg->header;
    publisher_3->publish(*image_out);

    // Stop the timer
    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print the execution time
    // RCLCPP_INFO(get_logger(), "Image processing took %lld milliseconds", duration);

  }

  void imageCallback4(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // Start the timer
    // auto start = std::chrono::steady_clock::now();
    // Convert sensor_msgs::Image to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Process the received image data here
    cv::Mat distorted_img = cv_ptr->image;

    float camera_matrix_data[3][3] = {{1280.329986, 0.000000, 1003.013096}, 
                                       {0.000000, 1281.692825, 768.736393}, 
                                       {0.000000, 0.000000, 1.000000}};

    cv::Mat camera_matrix(3, 3, CV_32F, camera_matrix_data);

    float distortion_coefficients_data[5] = {-0.210138, 0.076565, -0.000310, 0.001538, 0.000000};
    cv::Mat distortion_coefficients(1, 5, CV_32F, distortion_coefficients_data);
    cv::Mat undistorted_img;
    cv::undistort(distorted_img, undistorted_img, camera_matrix, distortion_coefficients);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_2;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_3;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_4;
    // cv::imshow("view", distorted_img);
    // cv::waitKey(30);

    // cv::namedWindow("Undistorted image", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Undistorted image", 512, 300);
    // cv::imshow("Undistorted image", undistorted_img);
    // cv::waitKey(30);

    // Example: Print image dimensions 
    // RCLCPP_INFO(get_logger(), "Received image 4: width = %d, height = %d", msg->width, msg->height);

    // Publish the undistorted image
    sensor_msgs::msg::Image::SharedPtr image_out = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", undistorted_img).toImageMsg();
    image_out->header = msg->header;
    publisher_4->publish(*image_out);

    // Stop the timer
    // auto end = std::chrono::steady_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Print the execution time
    // RCLCPP_INFO(get_logger(), "Image processing took %lld milliseconds", duration);

  }

  rclcpp::CallbackGroup::SharedPtr cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_1;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_2;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_3;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_4;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_1;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_2;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_3;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_4;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ImageSubscriber>();
  rclcpp::spin(node);
}