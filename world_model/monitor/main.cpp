#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/string.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
class Monitor : public rclcpp::Node {
 using Scan=sensor_msgs::msg::LaserScan;
 rclcpp::Publisher<Scan>::SharedPtr clean_,attack_,unknown_;
 rclcpp::Publisher<std_msgs::msg::String>::SharedPtr events_;
 rclcpp::Subscription<Scan>::SharedPtr sub_;
 public:
 Monitor():Node("scan_provenance") {
  clean_=create_publisher<Scan>("/observed/clean_scan",50);
  attack_=create_publisher<Scan>("/observed/pulser_scan",50);
  unknown_=create_publisher<Scan>("/observed/unknown_scan",50);
  events_=create_publisher<std_msgs::msg::String>("/experiment/provenance",50);
  sub_=create_subscription<Scan>("/scan_slam",rclcpp::SensorDataQoS().keep_last(50),
   [this](Scan::ConstSharedPtr m,const rclcpp::MessageInfo &info){
    const auto &gid=info.get_rmw_message_info().publisher_gid;
    std::string origin="unknown";
    for(const auto &ep:get_publishers_info_by_topic("/scan_slam")){
     const auto &eg=ep.endpoint_gid();
     if(std::equal(eg.begin(),eg.end(),gid.data)){
      if(ep.node_name()=="_scan_interceptor")origin="pulser";
      else if(ep.node_name()=="world_model_experiment")origin="clean";
     }
    }
    if(origin=="pulser")attack_->publish(*m);
    else if(origin=="clean")clean_->publish(*m);
    else unknown_->publish(*m);
    std::ostringstream o;o<<"{\"origin\":\""<<origin<<"\",\"stamp_ns\":"
      <<(int64_t(m->header.stamp.sec)*1000000000LL+m->header.stamp.nanosec)<<",\"publisher_gid\":\"";
    for(size_t i=0;i<RMW_GID_STORAGE_SIZE;i++)o<<std::hex<<std::setw(2)<<std::setfill('0')<<int(gid.data[i]);
    o<<"\"}";std_msgs::msg::String event;event.data=o.str();events_->publish(event);
   });
 }
};
int main(int argc,char **argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<Monitor>());rclcpp::shutdown();}
