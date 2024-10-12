# Learning Stable Imitation Policies with Waypoints

## Overview

This repository is the implementation code of the paper "Single-Shot Learning of Stable Dynamical Systems for Long-Horizon Manipulation Tasks"([arXiv](https://arxiv.org/abs/2410.01033)) by Alexandre St-Aubin, Amin Abyaneh, and Hsiu-Chin Lin at McGill University.

Imitation learning can be leveraged to tackle complex motion planning problems by training a policy to imitate an expert's behavior. We attempt to safely imitate *long-horizon* manipulation tasks from a single demonstration.

This repository builds on top of the following two publications:

* **SNDS** — A. Abyaneh, M. Sosa, H.-C. Lin. Globally stable neural imitation policies. International Conference on Robotics and Automation, 2024.

## Installation 
First, clone this repository on your device.
```
git clone https://github.com/Alestaubin/stable-imitation-policy-with-waypoints.git
cd stable-imitation-policy-with-waypoints
pip install -e .
```

Make sure you have Conda installed, or install it from [Anaconda Website](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Mac
Create and activate a new conda environment from the `environment-mac.yml` file.
```
conda env create -f environment-mac.yml
conda activate SIPWW
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

See [this](https://drive.google.com/drive/folders/16f09qTD5ZNinowQvYAyUhC49PSdX95H1?usp=sharing) drive for the hdf5 datasets. Each folder contains a different task. In each folder, you'll find the original dataset from `Libero` and the modified dataset as well as a video of the task. To be able to run these on your machine, you'll need to at least process the absolute paths using `lib/utils/preprocess_hdf5.py` (see `preprocess.sh`). 

Use the following commands to download the data.
```
cd data
gdown --folder https://drive.google.com/drive/folders/16f09qTD5ZNinowQvYAyUhC49PSdX95H1?usp=sharing
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

1. Download the dataset and move it to the data folder. Make sure to place the hdf5 file in a folder of the same name, e.g. `data/task1/task1.hdf5`.
```
cd data
mkdir name_of_task
cd path/to/task_dataset
mv task_dataset path/to/data/name_of_task/name_of_task.hdf5
```
2. Preprocess raw data and extract images (see `preprocess.sh`).

## Training

### config file
Create a config file with the following structure: 
```yaml
{
    "training":{
        "learner_type": "snds",
        "num_epochs": 10000,
        "demo": 1, # demo to train on in the dataset
        "device": "cpu",
        "segmentation": true # whether to segment the data into subgoals
    },
    "data":{
        "linear_policies": ["segment0_model_name", "segment1_model_name", "segment2_model_name"], 
        "angular_policies": null,
        "model_dir": "res/", # set to null if models are not trained yet
        "waypoints_dataset": "AWE_waypoints_01", # if null, will train on entire dataset
        "subgoals_dataset": "AWE_subgoals_01",
        "data_dir": "path/to/dataset/image_demo_local.hdf5" 
    },
    "simulation":{
        "playback": true, # whether to playback the policy in simulation
        "plot": false, # whether to output plots of the policy rollout
        "video_skip": 3, # number of frames to skip in the video
        "video_name": "rollout.mp4", # file to save the video to (if playback == True)
        "camera_names": ["agentview", "robot0_eye_in_hand"], 
        "write_video": true, 
        "slerp_steps": 50, # number of steps for the slerp algo (decrease if ori is not reached before subgoal)
        "perturb_step": null, # (list) step at which to inject perturbation 
        "perturb_ee_pos":null, # (list) position of the perturbation
        "reset_on_fail": true, # whether to reset to next subgoal on failure
        "noise_alpha": 0.01, # standard deviation of the gaussian noise added to the ee_pos feedback
        "grasp_tresh": 0.008, # threshold distance for subgoal to be considered a success (thus grasp)
        "release_tresh": 0.03 # different threshold for releasing
    },
    "data_processing":{
        "augment_rate": 4, # add 4 new datapoint for each original instance (data augmentation)
        "augment_alpha": 0.0025, # parameter for the augmentation
        "augment_distribution": "uniform", # uniform or normal 
        "normalize_magnitude": 0.25, # magnitude of the velocity vectors
        "clean": true # whether to remove out of distribution points 
    },
    "snds":{ # SNDS network parameters
        "fhat_layers": [256, 256, 256],
        "lpf_layers": [64, 64],
        "eps": 0.02,
        "alpha": 0.01,
        "relaxed": true
    },
    "testing":{ 
        "num_rollouts": 10, # number of rollouts for the simulation 
        "max_horizon": 1000, # max number of steps before cutoff
        "verbose": true 
    }
} 
```
Then, run the training script
```shell
python imitate-task.py --config path/to/config/file.json
```
The `k` trained models will be saved to `res/learner_type/`, where `k` is the number of subgoals in the task, `learner_type` is set in the config file (either snds or nn). 
## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting **pull requests** to us.

## Citation

Please use the following BibTeX formatted **citation**:
```
@misc{staubin2024singleshotlearningstabledynamical,
      title={Single-Shot Learning of Stable Dynamical Systems for Long-Horizon Manipulation Tasks}, 
      author={Alexandre St-Aubin and Amin Abyaneh and Hsiu-Chin Lin},
      year={2024},
      eprint={2410.01033},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.01033}, 
}
```

## Authors

* Alexandre St-Aubin
* Amin Abyaneh
* Hsiu-Chin Lin

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
