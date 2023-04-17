#!/bin/bash

# Create a Conda environment
conda create -n myenv python=3.10
conda activate myenv

# Install packages from requirements.txt using pip
pip3 install -r requirements.txt
cd ..
python3 demo.py