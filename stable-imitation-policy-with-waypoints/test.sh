#!/bin/bash


# Base directory for the configs
config_dir="config/6-cheese/run1"

# get a list of all files in config_dir
config_files=($(ls $config_dir))
# iterate over each file in config_files
for config_file in "${config_files[@]}"; do
    # get the path to the config file
    config_path="$config_dir/$config_file"
    # get the output file name
    output_file="${config_file%.json}.txt"
    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists, writing to ${config_file%.json}2.txt"
        output_file="${config_file%.json}2.txt"
    fi

    # run the Python script with the config file
    echo "Running imitation-task.py with $config_file"
    python imitate-task.py --config "$config_path" &> "$output_file"
done
