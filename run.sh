#!/bin/bash
# Run a Python script with Isaac Sim's Python, with conda stripped from the environment.
# (conda base is active in this shell; isaaclab.sh would otherwise pick conda's python.)
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda3 | paste -sd:)
exec ~/isaacsim/python.sh "$@"
