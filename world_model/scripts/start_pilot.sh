#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="${1:-pilot_new}"
container=world-model-sim
if ! docker image inspect turtlbot3_clone:latest >/dev/null 2>&1; then
  docker load -i "$root/artifacts/turtlebot3.tar.gz"
fi
if docker container inspect "$container" >/dev/null 2>&1; then
  docker start "$container" >/dev/null
else
  docker run -d --name "$container" --init --shm-size=512m \
    -e TURTLEBOT3_MODEL=burger -e ROS_DOMAIN_ID=74 -e ROS_LOCALHOST_ONLY=1 \
    -e LIBGL_ALWAYS_SOFTWARE=1 -e ALSOFT_DRIVERS=null \
    -e RECORD_OWNER_UID="$(id -u)" -e RECORD_OWNER_GID="$(id -g)" \
    -v "$root:/workspace" -w /workspace turtlbot3_clone:latest sleep infinity
fi
docker exec "$container" bash --noprofile --norc -c '
  source /opt/ros/humble/setup.bash
  cmake -S /workspace/world_model/monitor -B /tmp/world-model-monitor
  cmake --build /tmp/world-model-monitor -j2
'
docker exec "$container" bash --noprofile --norc -c '
  source /usr/share/gazebo/setup.sh
  source /opt/ros/humble/setup.bash
  export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models:${GAZEBO_MODEL_PATH:-}
  exec python3 /workspace/world_model/scripts/run_pilot.py --prefix "$1" --duration 60
' _ "$prefix"
