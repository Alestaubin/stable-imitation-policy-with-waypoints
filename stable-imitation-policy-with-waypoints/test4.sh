#!/bin/bash

#folders=("config/2-square" "config/4-bowl" "config/5-wine" "config/6-cheese" "config/6-cheese/nn.json")
folders=("config/6-cheese")
config_files=("nn.json") # "nn.json"
for folder in "${folders[@]}"; do
    # iterate over each file in config_files
    for config_file in "${config_files[@]}"; 
    do
        # get the path to the config file
        config_path="$folder/$config_file"
        # train and test 10 models
        for i in {0..9}; 
        do
            echo "Model: $config_path, iteration: $i"
            # get the output file name
            output_file="$folder/${config_file%.json}$i.txt"
            # check if the output file already exists
            if [ -f "$output_file" ]; then
                echo "Output file $output_file already exists"
                output_file="${output_file%.txt}1.txt"
                echo "Writing to $output_file"
            fi
            # run the Python script with the config file
            echo "Running imitation-task.py with $config_file"
            python imitate-task.py --config "$config_path" &> "$output_file"
        done
    done
done