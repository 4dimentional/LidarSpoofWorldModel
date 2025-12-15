from collections import deque
import statistics

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan


#node intercepts lidar scan stream applies temporal gate and beam smoothing then republishes
class TemporalScanInterceptor(Node):
    def __init__(self):
        #hidden node name to keep node unseen in graph tools
        super().__init__("_scan_interceptor")

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        #subscribe to scan topic and republish corrected frames back on scan
        self.sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)
        self.pub = self.create_publisher(LaserScan, "/scan", qos)

        #timing fields to learn real lidar period and drop bursts
        self.last_input_time = None
        self.nominal_period = None     #smoothed lidar period in seconds learned over time
        self.alpha = 0.2               #smoothing factor for timing model to follow slow drift
        self.last_forwarded_stamp = None  #tracks last forwarded stamp to avoid echo
        self.last_input_stamp = None   #tracks previous incoming stamp for burst filter
        self.min_period_sec = 0.08     #reject frames faster than 12 hz

        #window of recent scans for median based adjustments
        self.window_size = 5
        self.scan_window = deque(maxlen=self.window_size)

        #publish timing tracking
        self.last_publish_time = None

        self.get_logger().info("Temporal scan interceptor active on /scan.")

    #pulls each beam toward median baseline from recent scans to crush high frequency noise
    def beam_augmentation(self, ranges, window, msg):

        if len(window) < 2:
            return

        #drop current scan from history so baseline uses only prior frames
        history = list(window)
        if history and history[-1] is msg:
            history = history[:-1]
            
        if len(history) < 2:
            return

        r_min = msg.range_min if msg.range_min > 0.0 else 0.0
        r_max = msg.range_max if msg.range_max > 0.0 else float("inf")

        span = min(len(ranges), *(len(s.ranges) for s in history))
        if span == 0:
            return

        #compute per beam median baseline and median absolute deviation (mads) for adaptive clamp
        medians = []
        mads = []
        
        for i in range(span):
            vals = [scan.ranges[i] for scan in history]
            med = statistics.median(vals)
            medians.append(med)
            mads.append(statistics.median(abs(v - med) for v in vals))

        pull = 0.8        #pulls toward history to damp spikes
        bias_scale = 0.9   #global compression to shift map subtly without high frequency noise

        for i in range(span):
            baseline = medians[i]
            current = ranges[i]

            #pull measurement toward baseline to fit DFT
            adjusted = baseline + (current - baseline) * (1.0 - pull)

            #clamp deviation using mad so spikes flatten while steady bias remains
            mad = mads[i]
            if mad > 0.0:
                max_dev = 1.5 * mad
                residual = max(min(adjusted - baseline, max_dev), -max_dev)
                adjusted = baseline + residual

            #apply global compression
            adjusted *= bias_scale

            if adjusted < r_min:
                adjusted = r_min
            elif adjusted > r_max:
                adjusted = r_max

            ranges[i] = adjusted

    def scan_callback(self, msg: LaserScan):
        now = self.get_clock().now()
        stamp_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        stamp_float = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        #drop messages we just published to avoid feedback loop
        if self.last_forwarded_stamp == stamp_key:
            return

        #drop overly fast bursts and keep one frame per learned period
        if (self.last_input_stamp is not None and stamp_float - self.last_input_stamp < self.min_period_sec):
            return
        self.last_input_stamp = stamp_float

        #store this scan in sliding window for smoothing context
        self.scan_window.append(msg)

        #update timing model with current arrival spacing
        if self.last_input_time is not None:
            dt = (now - self.last_input_time).nanoseconds / 1e9

            #only trust deltas that match typical lidar rates between 2 and 10 hz
            if 0.1 <= dt <= 0.4:
                if self.nominal_period is None:
                    self.nominal_period = dt
                else:
                    self.nominal_period = (self.alpha * dt + (1.0 - self.alpha) * self.nominal_period)

        self.last_input_time = now

        #if timing unknown publish first frames one to one
        if self.nominal_period is None:
            self.publish_corrected(msg, now)
            return

        #publish gated by timing model so output matches real cadence
        if self.last_publish_time is None:
            self.publish_corrected(msg, now)
            return

        elapsed = (now - self.last_publish_time).nanoseconds / 1e9

        #only publish when elapsed time meets learned period
        if elapsed >= self.nominal_period:
            self.publish_corrected(msg, now)

    def publish_corrected(self, msg: LaserScan, now):
        #must have enough samples to smooth
        if len(self.scan_window) >= 2:
            #modify ranges using median based smoothing
            self.beam_augmentation(msg.ranges, self.scan_window, msg)

        #publish corrected frame on scan topic
        self.pub.publish(msg)
        self.last_forwarded_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.last_publish_time = now


def main(args=None):
    #run interceptor and exit cleanly on interrupt
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
