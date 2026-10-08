#!/usr/bin/env python3
"""Run inside an isolated simulation container; each episode has fresh Gazebo/SLAM."""
import argparse,subprocess,os,signal,time,json,fcntl
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
BASE=Path('/workspace/world_model')
def run_episode(name,seed,attack,duration):
 out=BASE/'data/processed'/name;raw=BASE/'data/raw'/name
 if out.exists() or raw.exists():raise FileExistsError(name)
 logs=BASE/'results'/name;logs.mkdir(parents=True,exist_ok=False)
 procs=[];files=[]
 def start(label,args):
  f=(logs/(label+'.log')).open('w');files.append(f)
  p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);procs.append(p);return p
 try:
  start('gazebo',['gzserver','/tmp/world-model.world','-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so'])
  start('robot',['ros2','launch','turtlebot3_gazebo','robot_state_publisher.launch.py','use_sim_time:=true'])
  spawn=start('spawn',['ros2','launch','turtlebot3_gazebo','spawn_turtlebot3.launch.py','x_pose:=-2.0','y_pose:=-0.5'])
  if spawn.wait(timeout=45)!=0:raise RuntimeError('Spawn failed')
  start('provenance',['/tmp/world-model-monitor/scan_provenance'])
  start('slam',['ros2','run','slam_toolbox','async_slam_toolbox_node','--ros-args','--params-file','/tmp/world-model-slam.yaml'])
  start('bag',['ros2','bag','record','--qos-profile-overrides-path',str(BASE/'configs/record_qos.yaml'),'-o',str(raw),'/scan','/scan_slam','/tf','/tf_static','/odom','/cmd_vel','/clock','/gazebo/model_states','/experiment/events','/experiment/provenance','/observed/clean_scan','/observed/pulser_scan','/observed/unknown_scan'])
  time.sleep(2)
  args=['python3',str(BASE/'scripts/experiment_node.py'),'--output',str(out),'--seed',str(seed),'--duration',str(duration)]
  if attack:args+=['--attack']
  p=start('experiment',args)
  if p.wait(timeout=duration*4+90)!=0:raise RuntimeError('Experiment failed; see '+str(logs))
 finally:
  for p in reversed(procs):
   if p.poll() is None:
    os.killpg(p.pid,signal.SIGINT)
    try:p.wait(timeout=12)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  for f in files:f.close()
  # Save ownership for all artifacts produced by this episode only.
  uid=int(os.environ['RECORD_OWNER_UID']);gid=int(os.environ['RECORD_OWNER_GID'])
  for root in [out,raw,logs,logs.parent/(name+'_pulser.log')]:
   if root.exists():
    for path in [root,*root.rglob('*')]:os.chown(path,uid,gid)
 print('Completed '+name,flush=True)
 time.sleep(2)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prefix',default='pilot_v1');p.add_argument('--duration',type=float,default=60);a=p.parse_args()
 lock=open('/tmp/world-model-pilot.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 tree=ET.parse('/opt/ros/humble/share/turtlebot3_gazebo/worlds/turtlebot3_world.world');world=tree.getroot().find('world')
 plugin=ET.SubElement(world,'plugin',name='world_model_ground_truth',filename='libgazebo_ros_state.so');ros=ET.SubElement(plugin,'ros');ET.SubElement(ros,'namespace').text='/gazebo';ET.SubElement(plugin,'update_rate').text='30.0';tree.write('/tmp/world-model.world')
 cfg=yaml.safe_load((BASE/'configs/slam_observed.yaml').read_text());params=cfg['/slam_toolbox']['ros__parameters'];params.update(scan_topic='/scan_slam',min_laser_range=.12,max_laser_range=3.5,use_sim_time=True)
 params={k:v for k,v in params.items() if v is not None};cfg={'slam_toolbox':{'ros__parameters':params}}
 Path('/tmp/world-model-slam.yaml').write_text(yaml.safe_dump(cfg))
 config_path=BASE/'configs'/f'{a.prefix}_slam.yaml';config_path.write_text(yaml.safe_dump(cfg));os.chown(config_path,int(os.environ['RECORD_OWNER_UID']),int(os.environ['RECORD_OWNER_GID']))
 manifest={'train':[],'validation':[],'test':[],'attack_test':[]}
 for i,split in enumerate(['train','train','train','validation','test','attack_test']):
  seed=i if split!='attack_test' else 4;name=f'{a.prefix}_{split}_{seed:02d}'
  if not (BASE/'data/processed'/name/'observations.npz').exists():run_episode(name,seed,split=='attack_test',a.duration)
  manifest[split].append(str(Path('data/processed')/name/'observations.npz'))
 path=BASE/'configs'/f'{a.prefix}_split.json';path.write_text(json.dumps(manifest,indent=2));os.chown(path,int(os.environ['RECORD_OWNER_UID']),int(os.environ['RECORD_OWNER_GID']))
