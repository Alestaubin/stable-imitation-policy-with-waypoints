#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
import random
import math
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Union
from matplotlib.patches import Ellipse, Patch


class PlotConfigs:
    """Hardcoded plot configurations.
    """

    COLORS = ["blue", "orange", "green", "purple", "brown"]
    FMTS = ['d--', 'o-', 's:', 'x-.', '*-', 'd--', 'o-']

    FIGURE_SIZE = (8, 8)
    FIGURE_DPI = 120
    POLICY_COLOR = 'grey'
    TRAJECTORY_COLOR = 'blue'
    ARROW_COLOR = "orange"
    ROLLOUT_COLOR = 'red'
    ANNOTATE_COLOR = 'black'
    TICKS_SIZE = 16
    LABEL_SIZE = 18
    LEGEND_SIZE = 18
    TITLE_SIZE = 18


def plot_trajectory(trajectory: np.ndarray, title: str = "", file_name: str = "",
                    save_dir: str = "", show_legends: bool = True, n_samples: int = 1000):
    """ Plot a given trajectory based on dimension.

    Args:
        trajectory (np.ndarray): Trajectory in form of a numpy array.
        title (str, optional): Title of the plot. Defaults to None.
        file_name(str, optional): Name of the plot file. Defaults to "".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".
    """

    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    x_1 = trajectory[:, 0]
    x_2 = trajectory[:, 1]

    plt.scatter(x_1, x_2, marker='o', s=3, color=PlotConfigs.TRAJECTORY_COLOR)
    plt.xlabel("X1", fontsize=PlotConfigs.LABEL_SIZE)
    plt.ylabel("X2", fontsize=PlotConfigs.LABEL_SIZE)

    plt.grid()

    start_points = np.array([trajectory[idx * n_samples] for  \
                             idx in range(int(len(trajectory) / n_samples))])
    goal_point = trajectory[-1]

    start_handle = plt.scatter(start_points[:, 0], start_points[:, 1], marker='x',
        color=PlotConfigs.ANNOTATE_COLOR, linewidth=3, s=120, label='Start')
    target_handle = plt.scatter(goal_point[0], goal_point[1], marker='*',
        color=PlotConfigs.ANNOTATE_COLOR, linewidth=2, s=250, label='Target')

    blue_dots = plt.Line2D([0], [0], color=PlotConfigs.TRAJECTORY_COLOR,
                           marker='o', label='Expert Demonstrations')

    if show_legends:
        plt.xlabel('X1', fontsize=PlotConfigs.LABEL_SIZE)
        plt.ylabel('X2', fontsize=PlotConfigs.LABEL_SIZE)
        plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='best',
            handles=[blue_dots, start_handle, target_handle])

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    if title is not None:
        plt.title(title, fontsize=PlotConfigs.TITLE_SIZE)

    if save_dir != "":
        name = file_name if file_name != "" else 'plot'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')
    else:
        plt.show()


def plot_gmm(trajectory: np.ndarray, means: List, covariances: List):
    """ This function plots the covariance and mean of the components of the GMM on the reference
        trajectory.

    Example:
        plot_gmm(trajectory=positions_py, means=gmm_sine.means_,
                covariances=gmm_sine.covariances_)

    Args:
        trajectory (np.ndarray): The reference trajectory.
        means (List): List of mean parameters for Gaussian models.
        covariances (List): List of covariance parameters for Gaussian models.
    """

    # generate the ellipses for gmm components

    ellipses = []
    for i in range(len(means)):
        v, w = np.linalg.eigh(covariances[i])
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        e = Ellipse(means[i], v[0], v[1], 180. + angle)
        ellipses.append(e)

    # plot the trajectory
    _, ax = plt.subplots(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)
    X1 = trajectory[:, 0]
    X2 = trajectory[:, 1]
    plt.scatter(X1, X2, marker='o', s=5)

    # plot the means
    for mean in means:
        plt.plot([mean[0]], [mean[1]], marker = 'x', markersize = 8, color='red')

    # plot the ellipses
    for ell in ellipses:
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.6)
        ell.set_facecolor(np.random.rand(3))

    x_min,x_max,y_min,y_max = find_limits(trajectory)
    ax.set_xlim(x_min * 0.9, x_max * 1.1)
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

    plt.grid()
    plt.xlabel('X1', fontsize=16)
    plt.ylabel('X2', fontsize=16)
    plt.show()


def plot_ds_stream(ds, trajectory: np.ndarray, title: str = None,
                   space_stretch: float = 0.1, stream_density: float = 1.0,
                   policy_density: int = 100, traj_density: int = 0.4,
                   file_name: str = "", save_dir: str = "", n_samples: int = 1000,
                   other_starts: List[np.ndarray] = None, n_reprod_trajs: int = 3,
                   show_legends: bool = True, show_rollouts: bool = True,
                   show_arrows: bool = False, save_rollouts: bool = False):
    """ Plot a policy for given a DS model and trajectories.

    Args:
        ds (PlanningPolicyInterface): A dynamical system for motion generation task.
        trajectory (np.ndarray): Input trajectory array (n_samples, dim).
        title (str, optional): Title of the plot. Defaults to None.
        space_stretch (float, optional): How much of the entire space to show in vector map.
            Defaults to 1.

        stream_density (float, optional): Density of policy streams. Defaults to 1.0.
        policy_density (int, optional): Density of on-trajectory policy arrows. Defaults to 10.
        traj_density (int, optional): Density of expert's trajectories. Defaults to 0.4.
        file_name(str, optional): Name of the plot file. Defaults to "".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".
        n_samples (int, optional): Number of samples in each demonstration. Defaults to 1000.
        other_starts (List[np.ndarray], optional): Other starting points to show stability.
            Defaults to None.
        n_reprod_trajs (int, optional): Number of trajectories to reproduce. Defaults to 10.
        show_legends (bool, optional): Opt to show the legends. Defaults to True.
    """

    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(trajectory)

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min - space_stretch, x_max + space_stretch])
    axes.set_ylim([y_min - space_stretch, y_max + space_stretch])

    # plot the trajectory
    start_points = np.array([trajectory[idx * n_samples] for  \
                             idx in range(int(len(trajectory) / n_samples))])
    goal_point = trajectory[-1]

    trimed_trajectory_idx = np.random.choice(a=len(trajectory),
                                             size=int(traj_density * len(trajectory)),
                                             replace=False)
    trimed_trajectory = np.array(trajectory[trimed_trajectory_idx])
    plt.scatter(trimed_trajectory[:, 0], trimed_trajectory[:, 1],
                color=PlotConfigs.TRAJECTORY_COLOR, marker='o',
                s=5, label='Expert Demonstrations')

    # generate the grid data
    x = np.linspace(x_min - space_stretch, x_max + space_stretch, policy_density)
    y = np.linspace(y_min - space_stretch, y_max + space_stretch, policy_density)
    X, Y = np.meshgrid(x, y)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    Z = np.apply_along_axis(lambda x: ds.predict(np.array([x])), 1, data)
    U, V = Z[:,:,0].reshape(policy_density, policy_density), \
        Z[:,:,1].reshape(policy_density, policy_density)

    # create streamplot
    plt.streamplot(X, Y, U, V, density=stream_density, color=PlotConfigs.POLICY_COLOR, linewidth=1)

    # on-trajectory policy-arrows
    if show_arrows:
        trimed_trajectory_idx = np.random.choice(a=len(trajectory),
                                                size=int(0.1 * traj_density * len(trajectory)),
                                                replace=False)
        trimed_trajectory = np.array(trajectory[trimed_trajectory_idx])

        res = ds.predict(trimed_trajectory)
        assert len(res.shape) == 2, "predict function is returning incorrect preds array!"

        for idx, point in enumerate(trimed_trajectory):
            norm = np.linalg.norm(res[idx])
            scale = 45 * norm

            plt.arrow(point[0], point[1], res[idx][0] / scale, res[idx][1] / scale,
                      color=PlotConfigs.ARROW_COLOR, width=0.004, label='Action')

    # on-trajectory policy-rollouts
    if show_rollouts:
        dt: float = 0.01

        if n_reprod_trajs > len(start_points):
            n_reprod_trajs = len(start_points)

        starts_idx = np.random.choice(a=len(start_points), size=n_reprod_trajs, replace=False)
        starts = start_points[starts_idx]
        starts = starts + other_starts if other_starts is not None else starts
        limit = np.linalg.norm([(x_max - x_min), (y_max - y_min)]) / 10
        for idx, start in enumerate(starts):
            simulated_traj: List[np.ndarray] = []
            simulated_traj.append(np.array([start]).reshape(1, 2))

            distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
            while  distance_to_target > limit  and len(simulated_traj) < 2e3:
                vel = ds.predict(simulated_traj[-1])
                simulated_traj.append(simulated_traj[-1] + dt * vel)
                distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)

            simulated_traj = np.array(simulated_traj)
            simulated_traj = simulated_traj.reshape(simulated_traj.shape[0],
                                                    simulated_traj.shape[2])

            if save_rollouts:
                name = file_name if file_name != "" else 'plot'
                np.save(os.path.join(save_dir, name, f'_rollout_{idx}'), simulated_traj)
            plt.plot(simulated_traj[:, 0], simulated_traj[:, 1],
                    color=PlotConfigs.ROLLOUT_COLOR, linewidth=2)


    # plot trajectory start and end
    start_handle = plt.scatter(start_points[:, 0], start_points[:, 1], marker='x',
        color=PlotConfigs.ANNOTATE_COLOR, linewidth=3, s=120, label='Start')
    target_handle = plt.scatter(goal_point[0], goal_point[1], marker='*',
        color=PlotConfigs.ANNOTATE_COLOR, linewidth=2, s=250, label='Target')

    green_arrows = plt.Line2D([0], [0], color=PlotConfigs.POLICY_COLOR,
                              linestyle='-', marker='>', label='Policy')
    red_arrows = plt.Line2D([0], [0], color=PlotConfigs.ROLLOUT_COLOR,
                            linestyle='-', marker='>', label='Reproduced')
    blue_dots = plt.Line2D([0], [0], color=PlotConfigs.TRAJECTORY_COLOR,
                           marker='o', label='Expert Demonstrations')

    if show_legends:
        plt.xlabel('X1', fontsize=PlotConfigs.LABEL_SIZE)
        plt.ylabel('X2', fontsize=PlotConfigs.LABEL_SIZE)

    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    # add legend with the custom handle
    if show_legends:
        plt.legend(fontsize=PlotConfigs.LEGEND_SIZE, loc='upper right',
            handles=[green_arrows, red_arrows, blue_dots, start_handle, target_handle])

    if title is not None:
        plt.title(title, fontsize=PlotConfigs.TITLE_SIZE)

    if save_dir != "":
        name = file_name if file_name != "" else 'plot'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')
    else:
        plt.show()


def multi_curve_plot_errorband(xs: Union[List[str], np.ndarray], y_means: List[np.ndarray],
        y_vars: List[np.ndarray], legends: List[str] = None, xlabel: str = "X",
        std_exaggeration: float = 1.0, ylabel: str = "Y",
        file_name: str = "", save_dir: str = "", use_boxes: bool = True,
        column_space: float = 20, inter_column_space: float = 2, log: bool = False):
    """ Plot multiple curves with errorbands.

    # TODO: Switch to datasamples instead of mean/var composition.
    # TODO: Messy function close to the deadline! Refactor later.

    Args:
        xs (List[str] or np.ndarray): Values for the xaxis.
        y_means (List[np.ndarray]): Mean values for the yaxis.
        y_varboths (List[np.ndarray]): Variance of yaxis.
        legends (List[str], optional): Legends corresponding to y_means. Defaults to None.
        xlabel (str, optional): xaxis label. Defaults to "X".
        ylabel (str, optional): yaxis label. Defaults to "Y".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".

        file_name(str, optional): Name of the plot file. Defaults to "".
    """

    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)
    axes = plt.gca()

    idx: int = 0
    violins: List = []
    for y_mean, y_var in zip(y_means, y_vars):
        if not use_boxes:
            plt.errorbar(x=xs, y=y_mean, yerr=y_var,
                color=PlotConfigs.COLORS[idx], label=legends[idx],
                fmt=PlotConfigs.FMTS[idx], capsize=5, elinewidth=2, markeredgewidth=3, linewidth=2)
        else:
            violins.append(axes.violinplot([np.random.normal(np.log(y_m) if log else y_m,
                                                             y_v * std_exaggeration,
                                                             size=100) \
                                            for y_m, y_v in zip(y_mean, y_var)],
                            positions=[(column_space * pos + idx * inter_column_space) for pos in range(1, len(xs) + 1)], widths=2.5, showmeans=True))

            for vp in violins[-1]['bodies']:
                vp.set_alpha(0.5)
                vp.set_linewidth(2)

        idx += 1

    axes.set_ylabel(ylabel, fontsize=PlotConfigs.LABEL_SIZE)
    axes.set_xlabel(xlabel, fontsize=PlotConfigs.LABEL_SIZE)

    if use_boxes:
        for x in [(column_space * (pos + 1/2) + (idx / 2) * inter_column_space) \
                            for pos in range(0, len(xs) + 1)]:
            axes.axvline(x, color = 'gray', linestyle='dashed', linewidth=1)

        axes.set_xticks([(column_space * pos + (idx / 2) * inter_column_space) \
                        for pos in range(1, len(xs) + 1)], labels=xs)
    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    if use_boxes:
        plt.grid(axis='y', linestyle='dashed')
        ["blue", "orange", "green", "purple", "brown"]
        legend_handles = [Patch(facecolor="blue", edgecolor='black'),
                          Patch(facecolor="orange", edgecolor='black'),
                          Patch(facecolor="green", edgecolor='black'),
                          Patch(facecolor="brown", edgecolor='black'),
                          Patch(facecolor="purple", edgecolor='black')
                          ]
        legend_labels = legends
        # plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=PlotConfigs.LEGEND_SIZE - 2)
    else:
        plt.grid(axis='both', linestyle='dashed')
        # plt.legend(loc='upper center', fontsize=PlotConfigs.LEGEND_SIZE - 2, ncol=5)

    if save_dir != "":
        name = file_name if file_name != "" else 'plot'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')

    plt.show()


def plot_performance_curves(file_path: str = "../../res/linear_direct_transfer.npy",
    keys: List[str] = ["linear"], num_execs: int = 100):
    """ Plotting the num_iters or time for transfer ds learning.

    Args:
        file_path (str, optional): Path of the data file. Defaults to
            "../../res/linear_direct_transfer.npy".
        num_execs (int, optional): Total number of executions. Defaults to 100.
    """

    # load and organize the data
    results = np.load(file_path, allow_pickle=True)

    for key in keys:
        reference_times = np.array([res[key]["reference_time"] for res in results])
        transfer_times = np.array([res[key]["transfer_time"] for res in results])
        partial_times = np.array([res[key]["partial_time"] for res in results])

        # plot the time performance
        xs = ['reference', 'transfer', 'partial']
        ys = [reference_times[:, 0], transfer_times[:, 0], partial_times[:, 0]]
        title = f'Evaluation of transfer retrain for {key} DS'
        xlabel = "Transfer policy"
        ylabel = "Optimization time (seconds)"

        fig = plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=150)
        axes = plt.gca()

        axes.boxplot(ys, meanline=True, showmeans=True)
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        axes.set_ylabel(ylabel, fontsize=10)
        axes.set_xlabel(xlabel, fontsize=10)

        axes.set_xticklabels(xs, fontsize=8)
        axes.set_title(title, fontsize=14)
        plt.savefig(f'time_performance_{key}.png')
        plt.show()

        # plot the number of iterations
        ys = [reference_times[:, 1], transfer_times[:, 1], partial_times[:, 1]]
        ylabel = "Optimization iterations"

        fig = plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)
        axes = plt.gca()

        axes.boxplot(ys, meanline=True, showmeans=True)
        axes.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        axes.set_ylabel(ylabel, fontsize=10)
        axes.set_xlabel(xlabel, fontsize=10)

        axes.set_xticklabels(xs, fontsize=8)
        axes.set_title(title, fontsize=14)
        plt.savefig(f'num_iters_performance_{key}.png')
        plt.show()


def find_limits(trajectory):
    """ Find the trajectory limits.

    Args:
        trajectory (np.ndarray): The given trajectory for finding limitations. Can be 2 or
            3 dimensions.

    Raises:
        NotSupportedError: Dimensions more than 3 are invalid.

    Returns:
        Tuple: A tuple of limits based on the dimensions (4 or 6 elements)
    """

    dimension = trajectory.shape[1]
    if dimension == 2:
        x_min = np.min(trajectory[:, 0])
        y_min = np.min(trajectory[:, 1])
        x_max = np.max(trajectory[:, 0])
        y_max = np.max(trajectory[:, 1])
        return x_min, x_max, y_min, y_max

    else:
        raise NotImplementedError('Dimension not supported')


def plot_contours(lpf, trajectory, step_size: float = 0.001, save_dir: str = "",
                  file_name: str = "", color: str = 'Greens_r',
                  space_stretch: float = 0.1):
    """Heatmap of an LPF function given a certain range.

    Args:
        lpf (Funciton): The function to plot.
        range (np.ndarray, optional): Ranges on both x and y axis in order.
            Defaults to [-10, 10, -10, 10].
        save_dir(str, optional): Provide a save directory for the figure.
            Leave empty to skip saving.
        color (str, 'Greens_r): Choose the color palette.
        file_name(str, ""): Name of the file to save.
        step_size (float, 0.001): Step size for contours. Default to 1e-3.
    """
    # find trajectory limits
    x_min, x_max, y_min, y_max = find_limits(trajectory)
    x_min, x_max = x_min - space_stretch, x_max + space_stretch
    y_min, y_max = y_min - space_stretch, y_max + space_stretch

    # calibrate the axis
    plt.figure(figsize=PlotConfigs.FIGURE_SIZE, dpi=PlotConfigs.FIGURE_DPI)

    axes = plt.gca()
    axes.set_xlim([x_min, x_max])
    axes.set_ylim([y_min, y_max])

    plt.scatter(trajectory[:, 0], trajectory[:, 1], color=PlotConfigs.TRAJECTORY_COLOR,
                marker='o', s=5, label='Expert Demonstrations')

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    data = np.concatenate([X.reshape(-1,1), Y.reshape(-1,1)], axis=1)
    Z = np.apply_along_axis(lpf, 1, data).reshape(100, 100)

    if np.min(Z) == 0 and np.max(Z) == 0:
        print(f'Aborting LPF plot since the function is not trained properly! In most cases this means additional training is required. Consider retraining with more epochs.')
        return

    Z /= np.linalg.norm(Z)
    step = np.abs(step_size)

    plt.contour(X, Y, Z, cmap=color, levels=np.arange(np.min(Z), np.max(Z) + step, step))
    plt.colorbar()
    plt.tick_params(axis='both', which='both', labelsize=PlotConfigs.TICKS_SIZE)

    plt.xlabel('X1', fontsize=PlotConfigs.LABEL_SIZE)
    plt.ylabel('X2', fontsize=PlotConfigs.LABEL_SIZE)

    if save_dir != "":
        name = file_name if file_name != "" else 'plot'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, name), dpi=PlotConfigs.FIGURE_DPI, bbox_inches='tight')
    else:
        plt.show()
