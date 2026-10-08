#!/usr/bin/env python3
"""Isolated Gazebo scan relay and feedback controller; never publish onto /scan."""
import argparse, copy, json, math, time, subprocess, signal, hashlib
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.qos import qos_profile_sensor_data


def yaw(q):return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
def wrap(x):return math.atan2(math.sin(x),math.cos(x))

class Experiment(Node):
 def __init__(self,a):
  super().__init__('world_model_experiment',parameter_overrides=[Parameter('use_sim_time',value=True)])
  self.a=a;self.start=None;self.latest=None;self.gt=None;self.gt_time=None;self.cmd=None;self.pose=None;self.pending=[];self.rows=[];self.done=False;self.attacker=None;self.attack_log=None;self.delivered_by_stamp={};self.attack_gids=set();self.last_discovery=-1;self.received_counts={"clean":0,"pulser":0,"unknown":0}
  self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
  self.pub=self.create_publisher(LaserScan,'/scan_slam',20)
  self.events=self.create_publisher(String,'/experiment/events',20)
  self.control=self.create_publisher(Twist,'/cmd_vel',10)
  self.create_subscription(LaserScan,'/scan',self.scan,qos_profile_sensor_data)
  self.create_subscription(ModelStates,'/gazebo/model_states',self.truth,qos_profile_sensor_data)
  self.create_subscription(LaserScan,'/scan_slam',self.latest_scan,qos_profile_sensor_data)
  for topic,origin in [('/observed/clean_scan','clean'),('/observed/pulser_scan','pulser'),('/observed/unknown_scan','unknown')]:
   self.create_subscription(LaserScan,topic,lambda m,o=origin:self.delivered(m,o),50)
  self.create_timer(.1,self.tick)
 def now(self):return self.get_clock().now().nanoseconds/1e9
 def truth(self,m):
  if 'burger' in m.name:
   p=m.pose[m.name.index('burger')];self.gt=[p.position.x,p.position.y,yaw(p.orientation)];self.gt_time=self.now()
 def latest_scan(self,m):self.latest=m
 def delivered(self,m,origin):
  stamp=m.header.stamp.sec*10**9+m.header.stamp.nanosec
  self.received_counts[origin]+=1
  self.delivered_by_stamp[stamp]=(list(m.ranges),origin)
 def scan(self,m):
  t=m.header.stamp.sec+m.header.stamp.nanosec/1e9
  if self.start is None:
   if self.gt is None:return
   self.start=t
  elapsed=t-self.start
  active=self.a.attack and 20<=elapsed<40
  scale=.9 if active else 1.
  out=copy.deepcopy(m)
  out.ranges=list(m.ranges)  # Unmodified relay; original PULSER process competes on /scan_slam.
  self.pub.publish(out)
  event={'scan_stamp_ns':int(m.header.stamp.sec)*10**9+m.header.stamp.nanosec,'sequence':len(self.rows)+len(self.pending),'elapsed':elapsed,'active':bool(active),'scale':scale,'source':'/scan','delivered':'/scan_slam'}
  self.events.publish(String(data=json.dumps(event)))
  if elapsed<self.a.duration:
   self.pending.append({'stamp':event['scan_stamp_ns'],'receipt':time.time_ns(),'clean':list(m.ranges),'scan':list(out.ranges),'min':m.range_min,'max':m.range_max,'cmd':self.cmd.copy() if self.cmd else None,'gt':self.gt.copy() if self.gt else None,'gt_time':self.gt_time,'elapsed':elapsed,'attack':bool(active),'scale':scale})
 def tick(self):
  if self.start is None:return
  elapsed=self.now()-self.start
  while self.pending and self.now()-self.pending[0]['stamp']/1e9>.6:
   row=self.pending.pop(0)
   selected=self.delivered_by_stamp.pop(row['stamp'],None)
   row['origin']=selected[1] if selected else 'unobserved'
   if selected:row['scan']=selected[0]
   try:
    t=self.buffer.lookup_transform('map','base_footprint',Time(nanoseconds=row['stamp']))
    row['pose']=[t.transform.translation.x,t.transform.translation.y,yaw(t.transform.rotation)]
   except TransformException:row['pose']=None
   self.rows.append(row)
  try:
   t=self.buffer.lookup_transform('map','base_footprint',Time());self.pose=[t.transform.translation.x,t.transform.translation.y,yaw(t.transform.rotation)]
  except TransformException:pass
  if self.a.attack and 20<=elapsed<40 and self.attacker is None:
   self.attack_log=(Path(self.a.output).parent.parent.parent/"results"/(Path(self.a.output).name+"_pulser.log")).open("w")
   self.attacker=subprocess.Popen(["python3","/workspace/src/TemporalScanBuster.py","--ros-args","-r","/scan:=/scan_slam"],stdout=self.attack_log,stderr=subprocess.STDOUT)
   self.events.publish(String(data=json.dumps({"kind":"attack_start","sim_seconds":self.now(),"implementation":"src/TemporalScanBuster.py","remap":"/scan:=/scan_slam"})))
  if self.attacker is not None and elapsed>=40 and self.attacker.poll() is None:
   self.attacker.send_signal(signal.SIGINT);self.attacker.wait(timeout=5)
   self.events.publish(String(data=json.dumps({"kind":"attack_stop","sim_seconds":self.now()})))
  cmd=Twist()
  if elapsed<self.a.duration and self.pose is not None and self.latest is not None:
   # Heading tracking closes the loop through SLAM; obstacle stop uses delivered LiDAR.
   desired=.65*math.sin(elapsed/9+self.a.seed*.65)
   cmd.angular.z=max(-.65,min(.65,1.1*wrap(desired-self.pose[2])))
   front=[r for i,r in enumerate(self.latest.ranges) if min(i,len(self.latest.ranges)-i)<25 and math.isfinite(r)]
   clearance=min(front) if front else 3.5
   cmd.linear.x=(.10+.025*math.sin(elapsed/7+self.a.seed)) if clearance>.5 else 0.
   if clearance<=.5:cmd.angular.z=.6
  self.control.publish(cmd);self.cmd=[cmd.linear.x,cmd.angular.z]
  if elapsed>=self.a.duration+1:
   self.done=True
 def save(self):
  rows=self.rows;n=len(rows)
  if n<32:raise RuntimeError('Insufficient scan data')
  def vec(k):return np.array([r[k] if r[k] is not None else [np.nan]*3 for r in rows])
  raw=np.array([r['scan'] for r in rows],dtype=np.float32);low=np.array([r['min'] for r in rows]);high=np.array([r['max'] for r in rows])
  valid=np.isfinite(raw)&(raw>=low[:,None])&(raw<=high[:,None])
  norm=np.where(valid,(raw-low[:,None])/(high-low)[:,None],0).astype(np.float32)
  pose=vec('pose');pv=np.isfinite(pose).all(1)
  output=Path(self.a.output);output.mkdir(parents=True,exist_ok=False)
  np.savez_compressed(output/'observations.npz',scan_ranges=raw,clean_scan_ranges=np.array([r['clean'] for r in rows],dtype=np.float32),scan_normalized=norm,scan_valid=valid,scan_stamp_ns=np.array([r['stamp'] for r in rows],dtype=np.int64),scan_bag_time_ns=np.array([r['receipt'] for r in rows],dtype=np.int64),scan_range_min=low,scan_range_max=high,slam_pose_xyyaw=pose,slam_pose_valid=pv,command_vw=np.array([r['cmd'] if r['cmd'] else [np.nan,np.nan] for r in rows]),command_observed=np.array([r['cmd'] is not None for r in rows]),ground_truth_xyyaw=vec('gt'),ground_truth_sim_seconds=np.array([r['gt_time'] for r in rows]),elapsed_seconds=np.array([r['elapsed'] for r in rows]),attack_active=np.array([r['attack'] for r in rows]),attack_scale=np.array([r['scale'] for r in rows]),scan_origin=np.array([r.get('origin','unobserved') for r in rows]))
  meta={'episode_id':output.name,'seed':self.a.seed,'duration':self.a.duration,'attack':self.a.attack,'attack_window':[20,40] if self.a.attack else None,'attack_scale':.9 if self.a.attack else 1.,'scan_count':n,'pose_valid_count':int(pv.sum()),'ground_truth':'Gazebo model_states burger world pose; latest received sample at scan callback; timestamp approximated by current /clock','scan_provenance':'original /scan relayed unchanged to /scan_slam; unmodified TemporalScanBuster remaps BOTH input and output to /scan_slam; publisher GIDs logged in events; one modeling row per original scan uses last received same-stamp scan, NOT a claim of SLAM acceptance', 'received_scan_counts':self.received_counts, 'attack_implementation_sha256':hashlib.sha256(Path('/workspace/src/TemporalScanBuster.py').read_bytes()).hexdigest(),'controller':'SLAM yaw feedback + delivered LiDAR obstacle stop','tf_semantics':'scan-time TF queried after 0.6 simulation seconds; unavailable poses masked','artifact_kind':'controlled_simulation_pilot; not physical robot evidence'}
  (output/'summary.json').write_text(json.dumps(meta,indent=2))
  print(json.dumps(meta),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--duration',type=float,default=60);p.add_argument('--attack',action='store_true');a=p.parse_args()
 rclpy.init();node=Experiment(a);deadline=time.monotonic()+a.duration*4+60
 try:
  while not node.done and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
  if not node.done:raise RuntimeError('Simulation timeout')
  node.save()
 finally:
  if node.attacker is not None and node.attacker.poll() is None:
   node.attacker.send_signal(signal.SIGINT);node.attacker.wait(timeout=5)
  if node.attack_log:node.attack_log.close()
  node.control.publish(Twist());node.destroy_node();rclpy.shutdown()
