#include <chrono>
#include <functional>
#include <memory>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class CameraPublisher : public rclcpp::Node
{
public:
    CameraPublisher()
        : Node("camera_publisher")
    {
        publisher_ =
            create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);

        camera_.open("http://192.168.1.4:8080/video");

        if (!camera_.isOpened()) {
            RCLCPP_ERROR(get_logger(), "无法打开摄像头");
        } else {
            RCLCPP_INFO(get_logger(), "摄像头已成功打开");
        }

        timer_ = create_wall_timer(
            33ms, std::bind(&CameraPublisher::publish_frame, this));
    }

private:
    void publish_frame()
    {
        cv::Mat frame;
        camera_ >> frame;

        if (frame.empty()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000, "没有读取到摄像头画面");
            return;
        }

        cv_bridge::CvImage image_message;
        image_message.header.stamp = now();
        image_message.encoding = sensor_msgs::image_encodings::BGR8;
        image_message.image = frame;

        publisher_->publish(*image_message.toImageMsg());
    }

    cv::VideoCapture camera_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CameraPublisher>());
    rclcpp::shutdown();
    return 0;
}
