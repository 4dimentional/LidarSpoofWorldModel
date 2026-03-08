
import argparse
import copy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan


class ScanFuzzerSafe(Node):
    #ros2 node subscribes to scan topic and fuzzes output

    def __init__(self,
                 #set variables
                 publish_rate_hz: float = 20.0,
                 offset_x: float = 0.0,
                 offset_y: float = 0.0,
                 yaw_offset: float = 0.0,
                 scale: float = 1.0,
                 publish_topic: str = '/scan_fuzz'):
        super().__init__('scan_fuzzer_safe')

        #store parameters as floats
        self.publish_rate_hz = float(publish_rate_hz)
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.yaw_offset = float(yaw_offset)
        self.scale = float(scale)
        self.publish_topic = publish_topic

        #storage for latest incoming scan and last published fuzzed scan
        self.latest_scan = None
        self.last_published = None

        #qos configuration
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50
        )

        #store deep copy of incoming scans so modifications keep original
        self.sub = self.create_subscription(LaserScan, '/scan', self._scan_cb, qos)

        #publisher to scan fuzz topic
        self.pub = self.create_publisher(LaserScan, self.publish_topic, qos)

        #timer
        #publish at configured rate
        period = 1.0 / max(0.0001, float(self.publish_rate_hz))
        self.timer = self.create_timer(period, self._timer_cb)
        self.get_logger().info(f"ScanFuzzerSafe publishing to {self.publish_topic} at {self.publish_rate_hz} Hz")

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = copy.deepcopy(msg)

    def _apply_point_transform(self, msg: LaserScan) -> LaserScan:
        #apply per point transforms
        #scale and rotate points by yaw offset
        #clamp to range bounds with small margin
        out = copy.deepcopy(msg)

        #precompute trig values for yaw rotation
        cos_y = math.cos(self.yaw_offset)
        sin_y = math.sin(self.yaw_offset)

        #provide defaults if range bounds missing
        rmin = getattr(msg, 'range_min', 0.0) or 0.0
        rmax = getattr(msg, 'range_max', 10.0) or 10.0
        angle = msg.angle_min

        for i in range(len(msg.ranges)):
            orig_r = msg.ranges[i]

            #preserve invalid or unset measurements
            if orig_r <= 0.0 or math.isinf(orig_r) or math.isnan(orig_r):
                new_r = orig_r
            else:
                #convert polar to cartesian
                x = orig_r * math.cos(angle)
                y = orig_r * math.sin(angle)

                #apply scale rotation and offset
                x *= self.scale
                y *= self.scale
                xr = cos_y * x - sin_y * y
                yr = sin_y * x + cos_y * y
                xt = xr + self.offset_x
                yt = yr + self.offset_y

                #convert back to range
                new_r = math.hypot(xt, yt)

                #clamp to safe bounds
                if new_r < rmin:
                    new_r = rmin
                if new_r > rmax:
                    new_r = rmax + 0.01

            out.ranges[i] = new_r
            angle += msg.angle_increment
        return out

    #construct and publish fuzzed scan
    def _timer_cb(self):
        #nothing to publish yet if no scan received
        if self.latest_scan is None:
            return
        
        #apply point level fuzzing
        out = self._apply_point_transform(self.latest_scan)

        #copy timestamp from real scan
        out.header.stamp = self.latest_scan.header.stamp

        #publish fuzzed message
        self.pub.publish(out)
        self.last_published = out


def main():
    #arguments to control fuzzing parameters
    parser = argparse.ArgumentParser(description='Safe scan fuzzer for /scan_fuzz')
    parser.add_argument('--publish-rate', type=float, default=20.0)
    parser.add_argument('--offset-x', type=float, default=0.0)
    parser.add_argument('--offset-y', type=float, default=0.0)
    parser.add_argument('--yaw-offset', type=float, default=0.0)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--topic', type=str, default='/scan_fuzz')
    args = parser.parse_args()

    rclpy.init()
    node = ScanFuzzerSafe(publish_rate_hz=args.publish_rate,
                          offset_x=args.offset_x,
                          offset_y=args.offset_y,
                          yaw_offset=args.yaw_offset,
                          scale=args.scale,
                          publish_topic=args.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
