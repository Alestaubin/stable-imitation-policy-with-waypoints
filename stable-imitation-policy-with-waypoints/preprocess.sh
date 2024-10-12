#!/bin/bash

# MODIFY THE DATA_FILE name
DATA_FILE="name_of_the_hdf5_file" # this file must be placed in a folder of the same name, e.g. data/task1/task1.hdf5
DATA_PATH="./data/$DATA_FOLDER"
MODIFIED_DATA_PATH="$DATA_PATH/demo_modified.hdf5" 
OUTPUT_NAME="image_demo_local.hdf5" 

###################################
####### Data Preprocessing ########
###################################

# this modifies the absolute paths in the hdf5 path (may need some tweaking)
python lib/utils/preprocess_hdf5.py -i "${DATA_PATH}/${DATA_FILE}.hdf5" -o "$MODIFIED_DATA_PATH" #--bddl_file "stable-imitation-policy-with-waypoints/stable-imitation-policy-with-waypoints/${DATA_PATH2}/${SCENARIO_NAME}.bddl"
# this converts absolute states to observations in the dataset (e.g. ee_vel, ee_pos, joint_angles, ...)
python lib/utils/dataset_states_to_obs.py --dataset "$MODIFIED_DATA_PATH" --done_mode 0 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --output_name "$OUTPUT_NAME" --exclude-next-obs
python lib/utils/dataset_extract_traj_plans.py --dataset "$DATA_PATH/$OUTPUT_NAME"

# Playback the dataset in simulation:
# NOTE: THIS IS NOT OPTIONAL, must playback sim to extract the euler angles
python sim/playback_robomimic_dataset.py --dataset "$DATA_PATH/$OUTPUT_NAME" --video_path "$DATA_PATH/demo.mp4" --start_idx 0 --end_idx 1
printf "Playback video saved to $DATA_PATH/demo.mp4\n"

###################################
########## AWE waypoints ##########
###################################
# 1. convert delta actions to absolute actions
python lib/utils/robomimic_convert_action.py --dataset "$DATA_PATH/$OUTPUT_NAME" --start_idx 0 --end_idx 1

# 2. extract waypoints
# NOTE: modify the following with the desired threshold
python lib/utils/waypoint_extraction.py --dataset "$DATA_PATH/$OUTPUT_NAME" --group_name "AWE_waypoints_01" --start_idx 0 --end_idx 1 --err_threshold 0.01
