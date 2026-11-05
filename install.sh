#!/bin/bash
# Register the jetbot_ramp tasks with Isaac Lab (editable install into Isaac Sim's Python).
# Isaac Lab's train/play CLI discovers them through the "isaaclab.tasks" entry point.
# Nothing inside ~/IsaacLab is modified. Undo with: ./run.sh -m pip uninstall -y jetbot_ramp
set -e
cd "$(dirname "$0")"
./run.sh -m pip install --no-deps --no-build-isolation -e .
./run.sh -c "import importlib.metadata as m; print([e.value for e in m.entry_points(group='isaaclab.tasks')])"
./run.sh -c "import jetbot_ramp, gymnasium as gym; print([k for k in gym.registry if 'JetbotRamp' in k])"
echo "installed"
