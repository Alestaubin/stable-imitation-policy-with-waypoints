# Learning Stable Imitation Policies with Waypoints

## Overview

Imitation learning can be leveraged to tackle complex motion planning problems by training a policy to imitate an expert's behavior. We attempt to safely imitate human demonstration from video data.

This repository builds on top of the following two publications:

* **SNDS** — A. Abyaneh, M. Sosa, H.-C. Lin. Globally stable neural imitation policies. International Conference on Robotics and Automation, 2024.

* **PLYDS** — A. Abyaneh and H.-C. Lin. Learning Lyapunov-stable polynomial dynamical systems through imitation. In 7th Annual
Conference on Robot Learning, 2023.

## Datasets

See [this](https://drive.google.com/drive/folders/1pUf4rRhM_E5hXXHynWmEhNyuBOxJXORY?usp=sharing) drive for the hdf5 datasets. Use the following commands to download the data.

```
cd data
gdown --folder https://drive.google.com/drive/folders/1pV-Lii52PF1djSSOAlLdCLpyS9zDAFIs?usp=drive_link
```

## Getting started

This section provides instructions on reproducibility and basic functionalities of the repository.

### Repository structure

To acquire a better understanding of the environment and features, you just need to clone the repository into your local machine. At first glance, the structure of the project appears below.

```bash
    ├── src          # Python source files of the project.
    ├── data         # Figures and other data.
    ├── res          # Resources like saved policies.
    ├── LICENSE
    ├── CONTRIBUTING.md
    └── README.md
```

### Dependencies and Conda

All the dependencies for this project are summarized as a Conda environment in [environment.yaml](environment.yaml). The following command should automatically install the entire set of dependencies.

```bash
conda env create -f environment.yaml
```

Before running the above, make sure you have Conda installed, or install it from [Anaconda Website](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

## Experimenting with SNDS

The [exp](exp/) folder contains most of the experiments and baselines comparisons. To prepare for running experiments, you need to navigate into the folder and use python to launch the file.

```bash
python3 nnds_training.py -nt snds -ms Sine -sp -nd 5 -ne 10000
```

The script utilizes the argparse library, so you can easily check the help instructions to understand their functionality and available command-line options using ```--help``` option.

## Experimenting with PLYDS

The [exp](exp/) folder contains most of the experiments and baselines comparisons. To prepare for running experiments, you need to navigate into the folder and grant Unix executable access to all the Python files or use python to launch them:

```bash
python plyds_learning.py
```


## Known Issues

* In some cases the training loss doesn't reduce when training the model, but the problem can be resolved by running the code again. We are trying to solve this problem at the moment.
* Note that in proper training, the loss should hover around 5e-3 and lower, otherwise the Lyapunov function might not be trained properly. Always allow sufficient epochs, 3-5k at least, to achieve this result, and also for learning rate scheduler to complete its task.
  ```cmd
  Train > 0.007863 | Test > 0.005087 | Best > (0.007863, 24) | LR > 0.00099
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
