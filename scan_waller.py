#!/usr/bin/env python3
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry



#  RAY INTERSECTION MATH
def ray_segment_intersect(px, py, dx, dy, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    denom = dx * vy - dy * vx
    if abs(denom) < 1e-8:
        return None
    t = ((x1 - px) * vy - (y1 - py) * vx) / denom
    u = ((x1 - px) * dy - (y1 - py) * dx) / denom
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


#  TEMPORAL INTERCEPTOR WITH WORLD-FIXED WALL
class TemporalScanInterceptor(Node):
    def __init__(self):
        super().__init__(
            "_scan_interceptor",
            start_parameter_services=False,
            automatically_declare_parameters_from_overrides=False,
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.pub = self.create_publisher(
            LaserScan, "/scan", qos
        )

        self.sub_odom = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos
        )
        self.robot_pose = None

        # Timing logic
        self.last_input_time = None
        self.nominal_period = None
        self.alpha = 0.2
        self.last_forwarded_stamp = None
        self.last_input_stamp = None
        self.min_period_sec = 0.08

        self.window_size = 5
        self.scan_window = deque(maxlen=self.window_size)

        self.last_publish_time = None

        # FIXED WALL 
        self.wall_x1 = 2.0
        self.wall_y1 = -1.0
        self.wall_x2 = 2.0
        self.wall_y2 =  1.0

        self.get_logger().info("Injecting fake world-wall...")


    #  STORE ODOMETRY
    def odom_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose


    #  WORLD-FIXED WALL
    def apply_world_wall(self, msg: LaserScan):
        if self.robot_pose is None:
            return

        rx = self.robot_pose.position.x
        ry = self.robot_pose.position.y
        q = self.robot_pose.orientation

        # yaw from quaternion
        yaw = 2.0 * math.atan2(q.z, q.w)

        angle = msg.angle_min

        for i in range(len(msg.ranges)):
            original_r = msg.ranges[i]

            dx = math.cos(yaw + angle)
            dy = math.sin(yaw + angle)

            # Check intersection
            t = ray_segment_intersect(
                rx, ry, dx, dy,
                self.wall_x1, self.wall_y1,
                self.wall_x2, self.wall_y2
            )

            # If intersection exists, override. Otherwise use the real scan.
            if t is not None:
                msg.ranges[i] = t
            else:
                msg.ranges[i] = original_r

            angle += msg.angle_increment


    #  SCAN CALLBACK
    def scan_callback(self, msg: LaserScan):
        now = self.get_clock().now()
        stamp_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        stamp_float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_forwarded_stamp == stamp_key:
            return

        if (
            self.last_input_stamp is not None
            and stamp_float - self.last_input_stamp < self.min_period_sec
        ):
            return
        self.last_input_stamp = stamp_float

        self.scan_window.append(msg)

        if self.last_input_time is not None:
            dt = (now - self.last_input_time).nanoseconds / 1e9
            if 0.1 <= dt <= 0.4:
                if self.nominal_period is None:
                    self.nominal_period = dt
                else:
                    self.nominal_period = (
                        self.alpha * dt
                        + (1 - self.alpha) * self.nominal_period
                    )

        self.last_input_time = now

        if self.nominal_period is None:
            self.publish_corrected(msg, now)
            return

        if self.last_publish_time is None:
            self.publish_corrected(msg, now)
            return

        elapsed = (now - self.last_publish_time).nanoseconds / 1e9
        if elapsed >= self.nominal_period:
            self.publish_corrected(msg, now)



    #  PUBLISH
    def publish_corrected(self, msg: LaserScan, now):
        self.apply_world_wall(msg)
        self.pub.publish(msg)
        self.last_forwarded_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.last_publish_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TemporalScanInterceptor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
