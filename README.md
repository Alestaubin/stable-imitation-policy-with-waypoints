# Learning Stable Imitation Policies with Waypoints

## Overview

Imitation learning can be leveraged to tackle complex motion planning problems by training a policy to imitate an expert's behavior. We attempt to safely imitate human demonstration from video data.

This repository builds on top of the following two publications:

* **SNDS** — A. Abyaneh, M. Sosa, H.-C. Lin. Globally stable neural imitation policies. International Conference on Robotics and Automation, 2024.

* **PLYDS** — A. Abyaneh and H.-C. Lin. Learning Lyapunov-stable polynomial dynamical systems through imitation. In 7th Annual
Conference on Robot Learning, 2023.

## Installation 
First, clone this repository on your device.
```
cd stable-imitation-policy-with-waypoints
```

Make sure you have Conda installed, or install it from [Anaconda Website](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Mac
Create and activate a new conda environment from the `environment-mac.yml` file.
```
conda env create -f environment-mac.yml
conda activate IL-waypoint
```
Install packages required for the simulation. 
```
# Install MuJoCo
pip install mujoco

# Install robosuite
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.4.1_libero
pip install -r requirements.txt
pip install -r requirements-extra.txt
pip install -e .

# Install BDDL
cd ..
git clone https://github.com/StanfordVL/bddl.git
cd bddl
pip install -e .

# Install LIBERO
cd ..
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .

# Install Robomimic
cd ..
git clone https://github.com/ARISE-Initiative/robomimic
cd robomimic
git checkout mimicplay-libero
pip install -e .

```
## Datasets

See [this](https://drive.google.com/drive/folders/1pUf4rRhM_E5hXXHynWmEhNyuBOxJXORY?usp=sharing) drive for the hdf5 datasets. Use the following commands to download the data.

```
cd data
gdown --folder https://drive.google.com/drive/folders/1pV-Lii52PF1djSSOAlLdCLpyS9zDAFIs?usp=drive_link
```
## Repository structure

To acquire a better understanding of the environment and features, you just need to clone the repository into your local machine. At first glance, the structure of the project appears below.

```bash
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── build/
├── environment-mac.yaml
├── environment.yaml
├── setup.py
├── stable-imitation-policy-with-waypoints # python package
│   ├── config/ # config files for imitate-task.py
│   ├── data/ # simulation datasets
│   ├── imitate-task.py
│   ├── lib/
│   │   ├── envs/
│   │   ├── learn_gmm_ds.py
│   │   ├── learn_nn_ds.py
│   │   ├── learn_ply_ds.py
│   │   ├── learn_rl_ds.py
│   │   ├── lipnet/
│   │   ├── nns/
│   │   ├── policy_interface.py
│   │   ├── rls
│   │   ├── seds/
│   │   ├── sos/
│   │   └── utils/
│   ├── res/ # saved policies
│   ├── sim
│   │   └── playback_robomimic.py
│   ├── src/
│   ├── train_waypoints.py
│   └── videos/
└── stable_imitation_policy_with_waypoints.egg-info
```
## Data Preprocessing
The model takes as input a panda arm task in the form of a hdf5 file. These tasks can be obtained from [Libero](https://libero-project.github.io/main.html). The original file must first be preprocessed before it is given as input to the model.

1. Download the dataset and move it to the data folder
```
cd data
mkdir name_of_task
cd path/to/task_dataset
mv task_dataset path/to/data/name_of_task.hdf5
```
2. Preprocess raw data and extract images
```
python lib/utils/preprocess_hdf5.py -i ./data/name_of_task/task_dataset.hdf5 -o ./data/name_of_task/demo_modified.hdf5 --workspace '/Users/alexst-aubin/SummerResearch24/V2' --libero_folder 'libero_90'

python lib/utils/dataset_states_to_obs.py --dataset './data/name_of_task/demo_modified.hdf5' --done_mode 0 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --output_name image_demo_local.hdf5 --exclude-next-obs
```
3. Extract end-effector trajectory for training
```
python lib/utils/dataset_extract_traj_plans.py --dataset 'data/name_of_task/image_demo_local.hdf5'
```
4. Convert delta actions to absolute actions for waypoint extraction
```
python lib/utils/robomimic_convert_action.py --dataset 'data/name_of_task/image_demo_local.hdf5'
```
5. Extract waypoints using [AWE](https://lucys0.github.io/awe/).
```
python lib/utils/waypoint_extraction.py --dataset 'data/name_of_task/image_demo_local.hdf5' --group_name 'AWE_waypoints_0025' --start_idx 0 --end_idx 49 --err_threshold 0.0025
```

## Training

### config file
Create a config file with the following structure: 
```json
{
    "training":{
        "learner_type": "snds",
        "num_epochs": 5000,
        "demos": [34],
        "device": "cpu"
    },
    "data":{
        "model_names": ["segment0_model_name", "segment1_model_name", "segment2_model_name"],
        "model_dir": "res/",
        "waypoints_dataset": "AWE_waypoints_0025",
        "subgoals_dataset": "AWE_subgoals_0025",
        "data_dir": "path/to/data/name_of_task/image_demo_local.hdf5"
    },
    "simulation":{
        "playback": true,
        "plot": true,
        "video_skip": 5,
        "multiplier": 1,
        "video_name": "a_very_cool_video.mp4",
        "camera_names": ["agentview", "robot0_eye_in_hand"]
    },
    "data_processing":{
        "augment_rate": 4,
        "augment_alpha": 0.0025,
        "augment_distribution": "uniform",
        "normalize_magnitude": 0.25,
        "clean": true
    },
    "snds":{
        "fhat_layers": [256, 256, 256],
        "lpf_layers": [64, 64],
        "eps": 0.02,
        "alpha": 0.01,
        "relaxed": true
    }
} 
```
Then, run the training script
```shell
python imitate-task.py --config path/to/config/file.json
```
## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting **pull requests** to us.

## Citation

Please use the following BibTeX formatted **citation** for PLYDS:
```
@inproceedings{abyaneh2023learning,
  title={Learning Lyapunov-Stable Polynomial Dynamical Systems Through Imitation},
  author={Abyaneh, Amin and Lin, Hsiu-Chin},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```
and SNDS:
```
@article{abyaneh2024globally,
  title={Globally Stable Neural Imitation Policies},
  author={Abyaneh, Amin and Guzm{\'a}n, Mariana Sosa and Lin, Hsiu-Chin},
  journal={arXiv preprint arXiv:2403.04118},
  year={2024}
}
```

## Authors

* Alexandre St-Aubin
* Amin Abyaneh
* Hsiu-Chin Lin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
