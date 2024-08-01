#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch

import numpy as np

from typing import Optional, Union

from lib.learn_nn_ds import NL_DS
from lib.learn_ply_ds import PLY_DS
from lib.utils.utils import time_stamp
from lib.utils.log_config import logger
from lib.utils.plot_tools import plot_ds_stream, plot_trajectory, plot_contours
from lib.utils.data_loader import load_hdf5_data


def waypoint_policy(learner_type: str,
                    waypoint_positions: np.ndarray,
                    waypoint_velocities: np.ndarray,
                    mode: str,
                    n_epochs: int,
                    plot: Optional[bool] = True,
                    model_name: Optional[str] = 'waypoint-test',
                    save_dir: str = 'res/',
                    gpu: Optional[bool] = True if torch.cuda.is_available() else False):

    """  Train a stable/unstable policy to learn a nonlinear dynamical system.

    Args:
        learner_type(str): Type of the nonlinear estimator, could be "nn" (unstable), "snds",
            "sdsef", or "plyds".

        mode (str): Mode between "train" and "test".
        n_epochs (int): Total number of epochs.
        plot (bool, Optional): Whether to plot trajectories and final ds or not.
            Default is True.

        model_name (str, Optional): Name of the model for save and load.
            Default is 'waypoint-test'.

        save_dir (str, Optional): Files will be saved in this directory if not None.
        gpu (bool, Optional):
    """

    # model and plot names
    name = f'{model_name}-{learner_type}-{time_stamp()}'

    # plot trajectory
    if plot:
        # NOTE: assuming only one trajectory
        plot_trajectory(waypoint_positions, file_name=name, save_dir=save_dir,
                        n_samples=waypoint_positions.shape[0])

    # NOTE: Choose between 4 different models, "plyds" and "snds" are ours.
    # "plyds" uses a polynomial and "snds" uses a neural representation.
    # "nn" is just unconstrained behavioral cloning.
    if learner_type == "plyds":
        model = PLY_DS(max_deg_ply=4, max_deg_lpf=2)
        model.fit(waypoint_positions, waypoint_velocities, optimizer="cvxpy", simplify_lpf=True)

    elif learner_type in ["snds", "nn"]:
        model = NL_DS(network=learner_type, data_dim=learner_type.shape[1], gpu=gpu)
        model.fit(waypoint_positions, waypoint_velocities, n_epochs=n_epochs)

    else:
        raise NotImplementedError(f'Learner type {learner_type} not available!')

    # plot the resulting ds and lpf
    if plot:
        plot_ds_stream(model, waypoint_positions, save_dir=save_dir, file_name=f'{name}-ds')

        if model.lpf() is not None:
            plot_contours(model.lpf, waypoint_positions, save_dir=save_dir,
                          file_name=f'{name}-lpf')

    # save the model
    model.save(model_name=name, dir=save_dir)


# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')

    # general params
    parser.add_argument('--learner-type', type=str, default="snds",
                        help='Policy learning method among snds, nn, plyds.')
    parser.add_argument('-m', '--mode', type=str, default="train",
                        help='Mode between train and test. Test mode only loads the model with the provided name.')

    parser.add_argument('-ne', '--num-epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=True,
                        help='Show extra plots of final result and trajectories.')

    parser.add_argument('-ts', '--test-size', type=float, default=0.01, help='Size of the validation set, not very important in this context.')

    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
                        help='Save the model in the res folder.')
    parser.add_argument('-sd', '--save-dir', type=str, default=os.path.join(os.pardir, 'res', 'nlds_policy'),
                        help='Optional destination for save/load.')
    parser.add_argument('-mn', '--model-name', type=str, default='test', help='Optional model name for saving.')

    parser.add_argument('-gp', '--gpu', type=bool, default=True, help='Enable or disable GPU support.')

    # SNDS params
    parser.add_argument('-rl', '--relaxed', type=bool, default=True, help='Relax asymptotic stability for SNDS.')
    parser.add_argument('-al', '--alpha', type=float, default=0.01, help='Exponential stability constant for SNDS as explained in the paper.')
    parser.add_argument('-ep', '--eps', type=float, default=0.01, help='Quadratic Lyapunov addition constant for SNDS as explained in the paper.')

    args = parser.parse_args()

    waypoint_policy(args.neural_tool, args.mode, args.motion_shape,
                        args.num_demonstrations, args.num_epochs, args.show_plots,
                        test_size=args.test_size, model_name=args.model_name,
                        save=args.save_model, save_dir=args.save_dir, gpu=args.gpu,
                        relaxed=args.relaxed, eps=args.eps, alpha=args.alpha)
