#!/bin/bash

# Check if the demo name is provided
if [ -z "$1" ]; then
  echo "Please provide the demo name as an argument."
  exit 1
fi

SCENARIO_NAME="$1"
DATA_PATH="./data/$SCENARIO_NAME"
DATA_PATH2="data/$SCENARIO_NAME"
MODIFIED_DATA_PATH="$DATA_PATH/demo_modified.hdf5"
OUTPUT_NAME="image_demo_local.hdf5"

######## Data Preprocessing ########
#python lib/utils/preprocess_hdf5.py -i "${DATA_PATH}/${SCENARIO_NAME}.hdf5" -o "$MODIFIED_DATA_PATH" #--bddl_file "stable-imitation-policy-with-waypoints/stable-imitation-policy-with-waypoints/${DATA_PATH2}/${SCENARIO_NAME}.bddl"
#python lib/utils/dataset_states_to_obs.py --dataset "$MODIFIED_DATA_PATH" --done_mode 0 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --output_name "$OUTPUT_NAME" --exclude-next-obs
#python lib/utils/dataset_extract_traj_plans.py --dataset "$DATA_PATH/$OUTPUT_NAME"

# Playback the dataset in simulation:
# NOTE: THIS IS NOT OPTIONAL, must playback sim to extract the euler angles
#python sim/playback_robomimic_dataset.py --dataset "$DATA_PATH/$OUTPUT_NAME" --video_path "$DATA_PATH/demo.mp4" --start_idx 0 --end_idx 1
#printf "Playback video saved to $DATA_PATH/demo.mp4\n"

########## AWE waypoints ##########
# convert delta actions to absolute actions
#python lib/utils/robomimic_convert_action.py --dataset "$DATA_PATH/$OUTPUT_NAME" --start_idx 0 --end_idx 1

# extract waypoints
python lib/utils/waypoint_extraction.py --dataset "$DATA_PATH/$OUTPUT_NAME" --group_name "AWE_waypoints_01" --start_idx 0 --end_idx 1 --err_threshold 0.01

########## Train imitation policy ##########
# NOTE: must set the correct parameters in the config file
python imitate-task.py --config config/square-snds.json

# IMPORTANT: set the correct path to the imitation policies in the config file before running the evaluation
# To evaluate the imitation policy, run the following command:
#python imitate-task.py --config config/controller.json