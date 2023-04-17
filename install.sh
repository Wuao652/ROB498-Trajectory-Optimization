#!/bin/bash

# Create a Conda environment
conda create -n myenv python=3.10
conda activate myenv

# install pybullet
git clone https://github.com/benelot/pybullet-gym.git
cd pybullet-gym
pip install -e .

cd ..
python3 demo.py