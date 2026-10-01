#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

class ColorDetector : public rclcpp::Node
{
public:
    ColorDetector()
        : Node("color_detector")
    {
        subscription_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw",
            10,
            std::bind(
                &ColorDetector::image_callback,
                this,
                std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "颜色识别节点已经启动");
    }

private:
    void draw_largest_block(
        const cv::Mat &mask,
        cv::Mat &result,
        const cv::Scalar &box_color,
        const std::string &label)
    {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(
            mask.clone(),
            contours,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_SIMPLE);

        double largest_area = 0.0;
        cv::Rect largest_rect;

        for (const auto &contour : contours) {
            double area = cv::contourArea(contour);

            if (area > 500.0 && area > largest_area) {
                largest_area = area;
                largest_rect = cv::boundingRect(contour);
            }
        }

        if (largest_area > 0.0) {
            cv::rectangle(result, largest_rect, box_color, 3);

            cv::putText(
                result,
                label,
                cv::Point(largest_rect.x, largest_rect.y - 10),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                box_color,
                2);
        }
    }

    void image_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr message)
    {
        cv::Mat frame;

        try {
            frame = cv_bridge::toCvCopy(
                message,
                sensor_msgs::image_encodings::BGR8)->image;
        } catch (const cv_bridge::Exception &error) {
            RCLCPP_ERROR(
                get_logger(),
                "图像转换失败：%s",
                error.what());
            return;
        }

        cv::Mat result = frame.clone();

        // 方法一：使用 HSV 阈值识别红色色块
        cv::Mat hsv;
        cv::Mat red_mask_1;
        cv::Mat red_mask_2;
        cv::Mat red_mask;

        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        cv::inRange(
            hsv,
            cv::Scalar(0, 100, 80),
            cv::Scalar(10, 255, 255),
            red_mask_1);

        cv::inRange(
            hsv,
            cv::Scalar(170, 100, 80),
            cv::Scalar(179, 255, 255),
            red_mask_2);

        cv::bitwise_or(red_mask_1, red_mask_2, red_mask);

        // 方法二：使用 BGR 通道差值识别蓝色色块
        std::vector<cv::Mat> channels;
        cv::Mat blue_green_difference;
        cv::Mat blue_red_difference;
        cv::Mat blue_green_mask;
        cv::Mat blue_red_mask;
        cv::Mat blue_mask;

        cv::split(frame, channels);

        cv::subtract(
            channels[0], channels[1], blue_green_difference);
        cv::subtract(
            channels[0], channels[2], blue_red_difference);

        cv::threshold(
            blue_green_difference,
            blue_green_mask,
            45,
            255,
            cv::THRESH_BINARY);

        cv::threshold(
            blue_red_difference,
            blue_red_mask,
            45,
            255,
            cv::THRESH_BINARY);

        cv::bitwise_and(
            blue_green_mask,
            blue_red_mask,
            blue_mask);

        // 去除小噪点并填补色块中的空洞
        cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(5, 5));

        cv::morphologyEx(
            red_mask,
            red_mask,
            cv::MORPH_OPEN,
            kernel);

        cv::morphologyEx(
            red_mask,
            red_mask,
            cv::MORPH_CLOSE,
            kernel);

        cv::morphologyEx(
            blue_mask,
            blue_mask,
            cv::MORPH_OPEN,
            kernel);

        cv::morphologyEx(
            blue_mask,
            blue_mask,
            cv::MORPH_CLOSE,
            kernel);

        draw_largest_block(
            red_mask,
            result,
            cv::Scalar(0, 0, 255),
            "Red block");

        draw_largest_block(
            blue_mask,
            result,
            cv::Scalar(255, 0, 0),
            "Blue block");

        cv::imshow("Color block detection", result);
        cv::imshow("Red HSV threshold", red_mask);
        cv::imshow("Blue channel difference", blue_mask);
        cv::waitKey(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ColorDetector>());
    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}
