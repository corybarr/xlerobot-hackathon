#!/usr/bin/env bash
# Record demos for one discrete skill on a single SO-101 arm pair.
#
# Usage:  ./scripts/record_skill.sh <skill_name> [num_episodes]
# Env:    HF_USER, FOLLOWER_PORT, LEADER_PORT, FRONT_CAM_IDX, HAND_CAM_IDX, SIDE_CAM_IDX
#
# Pushes dataset to HF Hub at $HF_USER/xlerobot-<skill_name>.
# Picks task description + episode time from skills/skills.yaml.

set -euo pipefail

SKILL="${1:?usage: record_skill.sh <skill_name> [num_episodes]}"
NUM="${2:-50}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_USER="${HF_USER:-Globalmysterysnailrevolution}"
FOLLOWER_PORT="${FOLLOWER_PORT:-COM10}"
LEADER_PORT="${LEADER_PORT:-COM7}"
FRONT_CAM_IDX="${FRONT_CAM_IDX:-1}"
HAND_CAM_IDX="${HAND_CAM_IDX:-0}"
SIDE_CAM_IDX="${SIDE_CAM_IDX:-2}"

# Look up skill metadata from yaml
read -r TASK_DESC EP_TIME <<<"$(python -c "
import sys, yaml
skills = yaml.safe_load(open('${REPO_ROOT}/skills/skills.yaml'))
s = skills.get('${SKILL}')
if not s:
    print(f'Unknown skill: ${SKILL}. Known: ' + ', '.join(skills.keys()), file=sys.stderr)
    sys.exit(1)
print(s['description'], s.get('episode_time_s', 15))
")"

echo "Recording skill='${SKILL}' (${NUM} episodes, ${EP_TIME}s each)"
echo "Task description: ${TASK_DESC}"
echo "Dataset repo: ${HF_USER}/xlerobot-${SKILL}"
echo

CAMERAS='{"hand_cam":{"type":"opencv","index_or_path":'${HAND_CAM_IDX}',"width":640,"height":480,"fps":30},"front_cam":{"type":"opencv","index_or_path":'${FRONT_CAM_IDX}',"width":640,"height":480,"fps":30},"side_cam":{"type":"opencv","index_or_path":'${SIDE_CAM_IDX}',"width":640,"height":480,"fps":30}}'

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=arm_a_follower \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=arm_a_leader \
  --dataset.repo_id="${HF_USER}/xlerobot-${SKILL}" \
  --dataset.single_task="${TASK_DESC}" \
  --dataset.num_episodes="${NUM}" \
  --dataset.episode_time_s="${EP_TIME}" \
  --display_data=true
