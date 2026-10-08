#!/usr/bin/env python3
"""Offline ROS 2 bag -> timestamped arrays; run in the ROS Humble container."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from tf2_ros import Buffer, TransformException


def ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def frame(name):
    # The bundled robot publishes leading slashes, which TF2 lookup rejects.
    return name.lstrip('/')


def pose7(position, orientation):
    return [position.x, position.y, position.z,
            orientation.x, orientation.y, orientation.z, orientation.w]


def planar(p):
    x, y, _, qx, qy, qz, qw = p
    return [x, y, math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))]


def require_monotonic(values, label, strict=False):
    d = np.diff(np.asarray(values, dtype=np.int64))
    if np.any(d < 0) or (strict and np.any(d == 0)):
        raise ValueError(f'{label}: reset, out-of-order, or duplicate timestamps; split/inspect episodes first')


def normalize_ranges(ranges, minimum, maximum):
    minimum, maximum = np.asarray(minimum), np.asarray(maximum)
    if np.any(maximum <= minimum):
        raise ValueError('Invalid sensor range bounds')
    valid = np.isfinite(ranges) & (ranges >= minimum[:, None]) & (ranges <= maximum[:, None])
    normalized = np.zeros_like(ranges, dtype=np.float32)
    scaled = (ranges-minimum[:, None])/(maximum-minimum)[:, None]
    normalized[valid] = scaled[valid]
    return normalized, valid


def align_commands(scan_receipt, command_receipt, commands, max_age_ns):
    """Unstamped Twist: last received command, bounded age in BAG clock domain."""
    values = np.full((len(scan_receipt), 2), np.nan)
    observed = np.zeros(len(scan_receipt), dtype=bool)
    age = np.full(len(scan_receipt), -1, dtype=np.int64)
    if len(command_receipt):
        indices = np.searchsorted(command_receipt, scan_receipt, side='right')-1
        for i, j in enumerate(indices):
            if j >= 0:
                age[i] = int(scan_receipt[i])-int(command_receipt[j])
                if age[i] <= max_age_ns:
                    values[i] = commands[j]
                    observed[i] = True
    return values, observed, age


def read_bag(path, storage):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=storage),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    wanted = {'/scan', '/tf', '/tf_static', '/clock', '/odom', '/cmd_vel'}
    classes = {t: get_message(typ) for t, typ in types.items() if t in wanted}
    records = {t: [] for t in wanted}
    while reader.has_next():
        topic, data, receipt = reader.read_next()
        if topic in classes:
            records[topic].append((receipt, deserialize_message(data, classes[topic])))
    if not records['/scan']:
        raise ValueError('No /scan messages')
    return records, types


def extract(path, output, storage='sqlite3', target='map', base='base_footprint', command_max_age=.5):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if command_max_age < 0:
        raise ValueError('command-max-age must be nonnegative')
    records, types = read_bag(path, storage)
    scans = records['/scan']
    scan_ns = np.array([ns(m.header.stamp) for _, m in scans], dtype=np.int64)
    receipt_ns = np.array([t for t, _ in scans], dtype=np.int64)
    require_monotonic(scan_ns, 'scan header', strict=True)
    require_monotonic(receipt_ns, 'scan bag time')
    clock_ns = np.array([ns(m.clock) for _, m in records['/clock']], dtype=np.int64)
    require_monotonic(clock_ns, '/clock')
    sizes = {len(m.ranges) for _, m in scans}
    if len(sizes) != 1:
        raise ValueError('Beam count changes within episode')
    geometry = {(m.angle_min, m.angle_max, m.angle_increment, frame(m.header.frame_id)) for _, m in scans}
    if len(geometry) != 1:
        raise ValueError('Scan geometry/frame changes within episode')
    ranges = np.array([m.ranges for _, m in scans], dtype=np.float32)
    minimum = np.array([m.range_min for _, m in scans], dtype=np.float32)
    maximum = np.array([m.range_max for _, m in scans], dtype=np.float32)
    normalized, valid = normalize_ranges(ranges, minimum, maximum)
    transforms = []
    for topic in ['/tf_static', '/tf']:
        for received, msg in records[topic]:
            for t in msg.transforms:
                t.header.frame_id, t.child_frame_id = frame(t.header.frame_id), frame(t.child_frame_id)
                transforms.append((topic == '/tf_static', received, t))
    dynamic_ns = [ns(t.header.stamp) for static, _, t in transforms if not static]
    span = max([*scan_ns, *dynamic_ns])-min([*scan_ns, *dynamic_ns])
    buffer = Buffer(cache_time=Duration(nanoseconds=int(span)+10_000_000_000))
    for static, _, t in sorted(transforms, key=lambda item: ns(item[2].header.stamp)):
        if static:
            buffer.set_transform_static(t, 'bag')
        else:
            buffer.set_transform(t, 'bag')
    poses = np.full((len(scans), 3), np.nan)
    pose_valid = np.zeros(len(scans), dtype=bool)
    errors = []
    for i, stamp in enumerate(scan_ns):
        if stamp == 0:  # TF2 treats zero as "latest", not exact epoch zero.
            errors.append({'index': i, 'reason': 'zero timestamp cannot request exact TF time'})
            continue
        try:
            t = buffer.lookup_transform(frame(target), frame(base), Time(nanoseconds=int(stamp)))
            poses[i] = planar(pose7(t.transform.translation, t.transform.rotation))
            pose_valid[i] = True
        except TransformException as exc:
            errors.append({'index': i, 'reason': str(exc)})
    delta = np.full_like(poses, np.nan)
    delta_valid = np.zeros(len(scans), dtype=bool)
    for i in range(len(scans)-1):
        if pose_valid[i] and pose_valid[i+1]:
            delta[i] = poses[i+1]-poses[i]
            delta[i, 2] = math.atan2(math.sin(delta[i, 2]), math.cos(delta[i, 2]))
            delta_valid[i] = True
    cmd_time = np.array([t for t, _ in records['/cmd_vel']], dtype=np.int64)
    require_monotonic(cmd_time, 'command bag time')
    commands = np.array([[m.linear.x, m.angular.z] for _, m in records['/cmd_vel']], dtype=float).reshape(-1, 2)
    control, control_valid, control_age = align_commands(receipt_ns, cmd_time, commands, int(command_max_age*1e9))
    arrays = dict(
        scan_ranges=ranges, scan_normalized=normalized, scan_valid=valid,
        scan_stamp_ns=scan_ns, scan_bag_time_ns=receipt_ns,
        scan_frame=np.array([m.header.frame_id for _, m in scans]),
        scan_range_min=minimum, scan_range_max=maximum,
        scan_angle_min=np.array([m.angle_min for _, m in scans]),
        scan_angle_increment=np.array([m.angle_increment for _, m in scans]),
        scan_time_increment=np.array([m.time_increment for _, m in scans]),
        slam_pose_xyyaw=poses, slam_pose_valid=pose_valid,
        next_slam_delta_xyyaw=delta, next_slam_delta_valid=delta_valid,
        command_vw=control, command_observed=control_valid, command_age_ns=control_age,
        raw_command_bag_time_ns=cmd_time, raw_command_vw=commands,
        clock_bag_time_ns=np.array([t for t, _ in records['/clock']], dtype=np.int64), clock_sim_ns=clock_ns,
        tf_stamp_ns=np.array([ns(t.header.stamp) for _, _, t in transforms], dtype=np.int64),
        tf_bag_time_ns=np.array([r for _, r, _ in transforms], dtype=np.int64),
        tf_static=np.array([s for s, _, _ in transforms], dtype=bool),
        tf_parent=np.array([t.header.frame_id for _, _, t in transforms]),
        tf_child=np.array([t.child_frame_id for _, _, t in transforms]),
        tf_xyz_xyzw=np.array([pose7(t.transform.translation, t.transform.rotation) for _, _, t in transforms]).reshape(-1, 7),
        odom_stamp_ns=np.array([ns(m.header.stamp) for _, m in records['/odom']], dtype=np.int64),
        odom_bag_time_ns=np.array([r for r, _ in records['/odom']], dtype=np.int64),
        odom_xyz_xyzw=np.array([pose7(m.pose.pose.position, m.pose.pose.orientation) for _, m in records['/odom']]).reshape(-1, 7),
        odom_vw=np.array([[m.twist.twist.linear.x, m.twist.twist.angular.z] for _, m in records['/odom']]).reshape(-1, 2),
        odom_parent=np.array([m.header.frame_id for _, m in records['/odom']]),
        odom_child=np.array([m.child_frame_id for _, m in records['/odom']]),
    )
    summary = {
        'schema_version': 1, 'episode_id': path.name, 'source_bag': str(path),
        'source_metadata_sha256': hashlib.sha256((path/'metadata.yaml').read_bytes()).hexdigest(),
        'topics': types, 'message_counts': {t:len(v) for t,v in records.items()},
        'scan_count': len(scans), 'beams': ranges.shape[1],
        'scan_sim_duration_seconds': int(scan_ns[-1]-scan_ns[0])/1e9,
        'valid_range_fraction': float(valid.mean()), 'tf_pose_valid_count': int(pose_valid.sum()),
        'tf_failures': errors, 'tf_target': frame(target), 'tf_base': frame(base),
        'delta_definition': 'next minus current in target frame; yaw wrapped to [-pi, pi]; last row invalid',
        'command_observed_count': int(control_valid.sum()), 'command_max_age_bag_seconds': command_max_age,
        'command_alignment': 'last Twist received no later than scan BAG timestamp; missing/stale = NaN and false',
        'time_domains': 'Header/TF/clock use simulation time; bag times are recorder timestamps. Never subtract the two as latency.',
        'tf_semantics': 'offline timestamp interpolation using full episode; may use later TF samples, not an online causal estimator',
        'normalization': '(range-min)/(max-min); invalid normalized entries=0 with separate false mask; raw values preserved',
        'scan_provenance': 'not inferred from bag; same-topic attack messages cannot be labeled by this extractor',
        'ground_truth': 'not available; odometry and SLAM are not labeled ground truth',
        'array_shapes': {k:list(v.shape) for k,v in arrays.items()},
    }
    output.mkdir(parents=True)
    np.savez_compressed(output/'observations.npz', **arrays)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['episode_id','scan_count','beams','tf_pose_valid_count','command_observed_count']}, indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--storage', default='sqlite3')
    p.add_argument('--target-frame', default='map')
    p.add_argument('--base-frame', default='base_footprint')
    p.add_argument('--command-max-age', type=float, default=.5)
    a=p.parse_args()
    extract(a.bag, a.output, a.storage, a.target_frame, a.base_frame, a.command_max_age)


if __name__ == '__main__':
    main()
