import tempfile
import unittest
from pathlib import Path
import math

import numpy as np
import rosbag2_py
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from rclpy.serialization import serialize_message
from extract_bag import extract, align_commands, require_monotonic


class ExtractionTests(unittest.TestCase):
    def test_missing_stale_and_future_commands(self):
        v, mask, age = align_commands(np.array([10,20,30,40]), np.array([15,35]),
                                      np.array([[1.,2.],[3.,4.]]), 5)
        np.testing.assert_array_equal(mask, [False,True,False,True])
        self.assertTrue(np.isnan(v[0]).all() and np.isnan(v[2]).all())
        np.testing.assert_array_equal(v[1], [1,2])
        np.testing.assert_array_equal(age, [-1,5,15,5])

    def test_reset_and_duplicate_are_rejected(self):
        for stamps in ([1,3,2], [1,1,2]):
            with self.assertRaises(ValueError):
                require_monotonic(stamps, 'test', strict=True)

    def test_offline_tf_interpolation_masks_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bag';out=Path(tmp)/'out'
            w=rosbag2_py.SequentialWriter()
            w.open(rosbag2_py.StorageOptions(uri=str(path), storage_id='sqlite3'),
                   rosbag2_py.ConverterOptions('cdr','cdr'))
            for topic,typ in [('/tf','tf2_msgs/msg/TFMessage'),('/tf_static','tf2_msgs/msg/TFMessage'),('/scan','sensor_msgs/msg/LaserScan')]:
                w.create_topic(rosbag2_py.TopicMetadata(name=topic,type=typ,serialization_format='cdr'))
            static=TransformStamped();static.header.frame_id='/map';static.child_frame_id='/odom'
            static.transform.rotation.w=1.
            w.write('/tf_static',serialize_message(TFMessage(transforms=[static])),100)
            def transform(sec,x):
                t=TransformStamped();t.header.stamp.sec=sec;t.header.frame_id='/odom';t.child_frame_id='/base_footprint'
                t.transform.translation.x=x;t.transform.rotation.w=1.
                return t
            w.write('/tf',serialize_message(TFMessage(transforms=[transform(1,0.)])),101)
            for sec,stamp in [(2,102),(4,104)]:
                scan=LaserScan();scan.header.stamp.sec=sec;scan.header.frame_id='base_scan'
                scan.range_min=.1;scan.range_max=3.5;scan.ranges=[1.,float('inf'),float('nan'),.05,4.]
                w.write('/scan',serialize_message(scan),stamp)
            w.write('/tf',serialize_message(TFMessage(transforms=[transform(3,2.)])),103)
            del w
            result=extract(path,out)
            a=np.load(out/'observations.npz',allow_pickle=False)
            np.testing.assert_allclose(a['slam_pose_xyyaw'][0], [1,0,0])
            np.testing.assert_array_equal(a['slam_pose_valid'],[True,False])
            self.assertTrue(np.isnan(a['slam_pose_xyyaw'][1]).all())
            np.testing.assert_array_equal(a['scan_valid'][0],[True,False,False,False,False])
            self.assertTrue(np.isinf(a['scan_ranges'][0,1]))
            self.assertTrue(np.isnan(a['scan_ranges'][0,2]))
            self.assertFalse(a['command_observed'].any())
            self.assertTrue(np.isnan(a['command_vw']).all())
            self.assertFalse(a['next_slam_delta_valid'].any())
            with self.assertRaises(FileExistsError):extract(path,out)


if __name__=='__main__':unittest.main()
