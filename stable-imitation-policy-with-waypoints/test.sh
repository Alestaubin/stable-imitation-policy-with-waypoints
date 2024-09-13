#!/bin/bash

# List of config files
config_files=(
    "kitchen1-nn-pert-noise.json"
    "kitchen1-nn-pert.json"
    "kitchen1-nn-seg-pert-noise.json"
    "kitchen1-nn-seg-pert.json"
    "kitchen1-nn-seg-way-pert-noise.json"
    "kitchen1-nn-seg-way-pert.json"
    "kitchen1-nn-seg-way.json"
    "kitchen1-nn-seg.json"
    "kitchen1-nn.json"
    "kitchen1-snds-pert-noise.json"
    "kitchen1-snds-pert.json"
    "kitchen1-snds-seg-pert-noise.json"
    "kitchen1-snds-seg-pert.json"
    "kitchen1-snds-seg-way-pert-noise.json"
    "kitchen1-snds-seg-way-pert.json"
    "kitchen1-snds-seg-way.json"
    "kitchen1-snds-seg.json"
)

# Base directory for the configs
config_dir="config"

# Iterate over each config file and run the Python script
for config_file in "${config_files[@]}"; do
    config_path="$config_dir/$config_file"
    output_file="${config_file%.json}.txt"

    echo "Running imitation-task.py with $config_file"
    python imitate-task.py --config "$config_path" &> "$output_file"
done