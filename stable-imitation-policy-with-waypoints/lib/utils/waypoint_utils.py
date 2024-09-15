import numpy as np
import matplotlib.pyplot as plt
from lib.utils.log_config import logger
import h5py
import json
from lib.utils.plot_tools import plot_ds_2Dstream, plot_trajectory, plot_contours, plot_ds_3Dstream, plot_ds_3Dstream2

def scatter_waypoints(waypoint_position, waypoint_velocity, title, save_path=None, subgoals=None):
    """
    A function that plots the waypoints and their velocities in 3D space.
    """
    waypoint_positions_cpy = np.copy(waypoint_position)
    waypoint_velocities_cpy = 0.05 * np.copy(waypoint_velocity)

    for i in range(len(waypoint_velocities_cpy)-1):
        dist = np.linalg.norm(waypoint_positions_cpy[i+1]-waypoint_positions_cpy[i])
        #print(f"distance between waypoints {i} and {i+1}: {dist}")
        #print(f"norm of velocity {i}: {np.linalg.norm(waypoint_velocities_cpy[i])}")
        if np.linalg.norm(waypoint_velocities_cpy[i]) > dist:
            # normalize the velocity vector
            waypoint_velocities_cpy[i] = (waypoint_velocities_cpy[i] / np.linalg.norm(waypoint_velocities_cpy[i])) * dist

    x, y, z = waypoint_positions_cpy[:, 0], waypoint_positions_cpy[:, 1], waypoint_positions_cpy[:, 2]
    u, v, w = waypoint_velocities_cpy[:, 0], waypoint_velocities_cpy[:, 1], waypoint_velocities_cpy[:, 2]

    #print(f"waypoint_position: {waypoint_position}")
    #print(f"waypoint_velocity: {waypoint_velocity}")
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    #change the background color to white for all three panels
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Plot positions
    ax.scatter(x, y, z, color='blue', label='Positions')

    # Plot velocity vectors (arrows)
    ax.quiver(x, y, z, u, v, w, color='red', label='Velocities')

    #ax.text(x[0]+0.02, y[0], z[0], 'Start', color='black', fontsize=12, fontweight='normal', fontstyle='italic')
    ax.scatter(x[0], y[0], z[0], c='black', marker='o')
    #ax.text(x[-1]-0.02, y[-1], z[-1] , 'Goal', color='black', fontsize=12, fontweight='normal', fontstyle='italic')
    ax.scatter(x[-1], y[-1], z[-1], c='black', marker='o')
    if subgoals is not None:
        # add labels at the end of the segment for each subgoal
        for j,i in enumerate(subgoals[:-1]):
            #ax.text(x[i]+0.02, y[i], z[i], f'subgoal {j}', color='black', fontsize=10, fontweight='normal', fontstyle='italic')
            # add a ball marker at the end of the segment
            ax.scatter(x[i], y[i], z[i], c='black', marker='o')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.legend()

    #ax.set_title(title)

    # save the plot
    if save_path is None:
        fig.savefig(f'plots/waypoints/{title.replace(" ", "-")}.png')
    else:
        fig.savefig(f'{save_path}/{title.replace(" ", "-")}.png')
        

def plot_rollouts(data, policies, path, title, perturbation=True, perturbation_step=50, perturbation_vec=None, reset_after_subgoal=False):
    all_segments = []
    all_waypoints = []
    all_velocities = []
    subgoals = []

    if len(policies) == 1:
        print("Only one policy trained for the entire task.")
        # handle the case where there is only one policy trained for the entire task
        policies = [policies[0] for _ in range(3)]

    first = True
    action_num = 0

    for i, ds_policy in enumerate(policies):
        waypoint_position = data["subgoal_" + str(i)]["waypoint_position"]
        waypoint_velocity = data["subgoal_" + str(i)]["waypoint_linear_velocity"]
        waypoint_velocity = normalize_waypoints(waypoint_velocity, 1)
        all_velocities.append(waypoint_velocity)
        all_waypoints.append(waypoint_position)
        subgoals.append(len(waypoint_position)-1 + subgoals[-1] if i > 0 else len(waypoint_position)-1)
        
        if first:
            start_point = waypoint_position[0]
            this_segment = [start_point]
            first = False
        else:
            if reset_after_subgoal:
                # start from the last subgoal
                this_segment = [waypoint_position[0]]
            else:
                # start from the last point of the previous segment
                this_segment = [all_segments[-1][-1]]
        
        dt = 0.01
        goal_point = waypoint_position[-1].reshape(1, waypoint_position.shape[1])
        
        distance_to_target = np.linalg.norm(this_segment[-1] - goal_point)

        while distance_to_target > 0.01 and len(this_segment) < 1000 or len(this_segment) < 10:
            if action_num == perturbation_step and perturbation:
                print(f'Adding perturbation at step {action_num}')
                this_segment.append(perturbation_vec)
            curr_ee_pos = np.array(this_segment[-1]).reshape(1,3)
            
            #if len(this_segment) % 25 == 0:
                #print("current ee pos: ", curr_ee_pos)

            vel = ds_policy.predict(curr_ee_pos)

            if not isinstance(dt, np.ndarray):
                dt = np.array(dt, dtype=np.float32)
            if not isinstance(vel, np.ndarray):
                vel = np.array(vel, dtype=np.float32)
            next_ee_pos = this_segment[-1] + dt * vel

            if next_ee_pos.shape[0] == 1: # To fix a very annoying bug
                next_ee_pos = next_ee_pos.squeeze()

            this_segment.append(next_ee_pos)

            distance_to_target = np.linalg.norm(this_segment[-1] - goal_point)
            action_num += 1

        this_segment = np.array(this_segment)
        this_segment = this_segment.squeeze()
        all_segments.append(this_segment)
        print(f'segment finished with distance to target: {distance_to_target}')

    all_waypoints = np.array(all_waypoints)
    all_segments = np.array(all_segments)
    print (all_segments.shape)
    print (all_segments[0].shape, all_segments[1].shape, all_segments[2].shape)
    print (perturbation_vec)
    all_velocities = np.array(all_velocities)

    scatter_waypoints(np.concatenate(all_waypoints), np.concatenate(all_velocities), 'AWE Waypoints with 0.01 treshold', path, subgoals=subgoals)

    # plot all segment and waypoints
    x0, y0, z0 = all_waypoints[0][:, 0], all_waypoints[0][:, 1], all_waypoints[0][:, 2]
    x0_segment, y0_segment, z0_segment = all_segments[0][:, 0], all_segments[0][:, 1], all_segments[0][:, 2]
    x1, y1, z1 = all_waypoints[1][:, 0], all_waypoints[1][:, 1], all_waypoints[1][:, 2]
    x1_segment, y1_segment, z1_segment = all_segments[1][:, 0], all_segments[1][:, 1], all_segments[1][:, 2]
    x2, y2, z2 = all_waypoints[2][:, 0], all_waypoints[2][:, 1], all_waypoints[2][:, 2]
    x2_segment, y2_segment, z2_segment = all_segments[2][:, 0], all_segments[2][:, 1], all_segments[2][:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #change the background color to white for all three panels
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.plot(x0, y0, z0, c='b', label='waypoints')
    ax.plot(x0_segment, y0_segment, z0_segment, c='r', label='segment')
    ax.plot(x1, y1, z1, c='b')
    ax.plot(x1_segment, y1_segment, z1_segment, c='r')
    ax.plot(x2, y2, z2, c='b')
    ax.plot(x2_segment, y2_segment, z2_segment, c='r',)
    #add labels at the end of the segment for each subgoal
    #ax.text(x0[-1]+0.015, y0[-1], z0[-1], 'subgoal 0', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #ax.text(x1[-1]-0.05, y1[-1]+0.01, z1[-1]-0.07, 'subgoal 1', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #ax.text(x2[-1]-0.04, y2[-1], z2[-1]+0.01, 'subgoal 2', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # add a ball marker at the end of the segment
    ax.scatter(x0[-1], y0[-1], z0[-1], c='black', marker='o')
    ax.scatter(x1[-1], y1[-1], z1[-1], c='black', marker='o')
    ax.scatter(x2[-1], y2[-1], z2[-1], c='black', marker='o')
    # add a ball marker at the start of the segment
    ax.scatter(x0[0], y0[0], z0[0], c='black', marker='o')
    #ax.text(x0[0], y0[0], z0[0]+0.03, 'start', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # add a ball marker at the point of perturbation
    if perturbation:
        ax.scatter(perturbation_vec[0], perturbation_vec[1], perturbation_vec[2], c='black', marker='o')
        #ax.text(perturbation_vec[0]-0.1, perturbation_vec[1], perturbation_vec[2], 'artificial perturbation', color='black', fontsize=9, fontweight='normal', fontstyle='italic')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.legend()
    #ax.set_title(f'{title} 3D Trajectory')
    fig.savefig(f'{path}/{title.replace(" ", "-")}-3D-entire-traj.png', dpi=1000)

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

    # X-Y projection
    axes[0].plot(x0, y0, c='b', label='waypoints')
    axes[0].plot(x0_segment, y0_segment, c='r', label='segment')
    axes[0].plot(x1, y1, c='b')
    axes[0].plot(x1_segment, y1_segment, c='r')
    axes[0].plot(x2, y2, c='b')
    axes[0].plot(x2_segment, y2_segment, c='r')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    #axes[0].set_title('X-Y Projection')
    #axes[0].legend()
    #axes[0].text(x0[-1]+0.03, y0[-1], 'subgoal 0', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[0].text(x1[-1]-0.02, y1[-1]+0.01, 'subgoal 1', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[0].text(x2[-1]-0.03, y2[-1]-0.02, 'subgoal 2', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    axes[0].scatter(x0[-1], y0[-1], c='black', marker='o')
    axes[0].scatter(x1[-1], y1[-1], c='black', marker='o')
    axes[0].scatter(x2[-1], y2[-1], c='black', marker='o')
    axes[0].scatter(x0[0], y0[0], c='black', marker='o')
    #axes[0].text(x0[0], y0[0]+0.005, 'start', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # add a ball marker at the point of perturbation
    if perturbation:
        axes[0].scatter(perturbation_vec[0], perturbation_vec[1], c='black', marker='o')
        #axes[0].text(perturbation_vec[0]-0.01, perturbation_vec[1]+0.02, 'artificial perturbation', color='black', fontsize=9, fontweight='normal', fontstyle='italic')

    # X-Z projection
    axes[1].plot(x0, z0, c='b', label='waypoints')
    axes[1].plot(x0_segment, z0_segment, c='r', label='segment')
    axes[1].plot(x1, z1, c='b')
    axes[1].plot(x1_segment, z1_segment, c='r')
    axes[1].plot(x2, z2, c='b')
    axes[1].plot(x2_segment, z2_segment, c='r')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    #axes[1].set_title('X-Z Projection')
    #axes[1].legend()
    #axes[1].text(x0[-1]+0.02, z0[-1], 'subgoal 0', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[1].text(x1[-1]-0.05, z1[-1]-0.005, 'subgoal 1', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[1].text(x2[-1]-0.05, z2[-1], 'subgoal 2', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    axes[1].scatter(x0[-1], z0[-1], c='black', marker='o')
    axes[1].scatter(x1[-1], z1[-1], c='black', marker='o')
    axes[1].scatter(x2[-1], z2[-1], c='black', marker='o')
    axes[1].scatter(x0[0], z0[0], c='black', marker='o')
    #axes[1].text(x0[0], z0[0]+0.005, 'start', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # add a ball marker at the point of perturbation
    if perturbation:
        axes[1].scatter(perturbation_vec[0], perturbation_vec[2], c='black', marker='o')
        #axes[1].text(perturbation_vec[0]+0.01, perturbation_vec[2]-0.01, 'artificial perturbation', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # Y-Z projection
    axes[2].plot(y0, z0, c='b', label='waypoints')
    axes[2].plot(y0_segment, z0_segment, c='r', label='segment')
    axes[2].plot(y1, z1, c='b')
    axes[2].plot(y1_segment, z1_segment, c='r')
    axes[2].plot(y2, z2, c='b')
    axes[2].plot(y2_segment, z2_segment, c='r')    
    axes[2].set_xlabel('y')
    axes[2].set_ylabel('z')
    #axes[2].set_title('Y-Z Projection')
    #axes[2].legend()
    #axes[2].text(y0[-1]+0.02, z0[-1], 'subgoal 0', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[2].text(y1[-1]-0.06, z1[-1], 'subgoal 1', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    #axes[2].text(y2[-1]-0.025, z2[-1]+0.0125, 'subgoal 2', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    axes[2].scatter(y0[-1], z0[-1], c='black', marker='o')
    axes[2].scatter(y1[-1], z1[-1], c='black', marker='o')
    axes[2].scatter(y2[-1], z2[-1], c='black', marker='o')
    axes[2].scatter(y0[0], z0[0], c='black', marker='o')
    #axes[2].text(y0[0]+0.01, z0[0], 'start', color='black', fontsize=9, fontweight='normal', fontstyle='italic')
    # add a ball marker at the point of perturbation
    if perturbation:
        axes[2].scatter(perturbation_vec[1], perturbation_vec[2], c='black', marker='o')
        #axes[2].text(perturbation_vec[1]-0.12, perturbation_vec[2], 'artificial perturbation', color='black', fontsize=9, fontweight='normal', fontstyle='italic')

    plt.suptitle(f'{title} Trajectory 2D Projections')
    fig2.savefig(f'{path}/{title.replace(" ", "-")}-2D-entire-traj-{i}.png', dpi=1000)





def normalize_waypoints(waypoint_velocity, magnitude):
    """
    A function that normalizes the waypoint velocities.
    """
    logger.info("Normalizing waypoint velocities...")
    # set the last waypoint velocity to zero
    waypoint_velocity[-1] = np.zeros_like(waypoint_velocity[-1])

    for i in range(waypoint_velocity.shape[0]):
        norm = np.linalg.norm(waypoint_velocity[i])

        if norm > 0:
            normalized_velocity = (waypoint_velocity[i] / norm) * magnitude
        else:
            normalized_velocity = np.zeros_like(waypoint_velocity[i])  # Handle zero vector case
        waypoint_velocity[i] = normalized_velocity
    return waypoint_velocity

def augment_data(waypoint_positions, waypoint_velocities, alpha=0.01, augment_rate=5, distribution='normal'):
    """Augment the data by adding Gaussian noise to the waypoints."""

    new_positions = []
    new_velocities = []

    for i in range(len(waypoint_positions)-1):
        # add the original point
        new_positions.append(waypoint_positions[i])
        new_velocities.append(waypoint_velocities[i])
        # for each original point apart from the last one, generate augment_rate new points
        for _ in range(augment_rate):
            if distribution == 'normal':
                noise = np.random.normal(0, alpha, waypoint_positions[i].shape)
            elif distribution == 'uniform':
                noise = np.random.uniform(-alpha, alpha, waypoint_positions[i].shape)
            else:
                raise ValueError(f"Unknown distribution type: {distribution}")
            new_position = waypoint_positions[i] + noise
            new_positions.append(new_position)
            velocity_norm = np.linalg.norm(waypoint_velocities[i])
            old_norm = np.linalg.norm(new_position - waypoint_positions[i+1])
            new_velocities.append(( waypoint_positions[i+1] - new_position )/old_norm * velocity_norm)
    
    # add the last point
    new_positions.append(waypoint_positions[-1])
    new_velocities.append(waypoint_velocities[-1])

    # Convert to numpy arrays
    augmented_positions = np.array(new_positions)
    augmented_velocities = np.array(new_velocities)

    #print(f"augmented_velocities: {augmented_velocities}") 
    return augmented_positions, augmented_velocities

def clean_waypoints(waypoint_positions, waypoint_velocities):
    """
    A function that removes any ood waypoints.
    """
    if len(waypoint_positions) < 3:
        return waypoint_positions, waypoint_velocities

    for i in range(1, len(waypoint_positions)-2): # last waypoint has zero velocity, 
        # if adjacent waypoints have velocity vectors that are too different, check which one is the outlier
        # compare angle between vectors
        if angle_between(waypoint_velocities[i], waypoint_velocities[i+1]) > np.pi/2 and angle_between(waypoint_velocities[i], waypoint_velocities[i-1])  > np.pi/2:
            waypoint_positions = np.delete(waypoint_positions, i, axis=0)
            waypoint_velocities = np.delete(waypoint_velocities, i, axis=0)
    
    #check first and second to last waypoint 
    if angle_between(waypoint_velocities[0], waypoint_velocities[1]) > np.pi/2:
        waypoint_positions = np.delete(waypoint_positions, 0, axis=0)
        waypoint_velocities = np.delete(waypoint_velocities, 0, axis=0)
    if angle_between(waypoint_velocities[-2], waypoint_velocities[-1]) > np.pi/2:
        waypoint_positions = np.delete(waypoint_positions, -1, axis=0)
        waypoint_velocities = np.delete(waypoint_velocities, -1, axis=0)
    
    return waypoint_positions, waypoint_velocities

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def scatter_ee_pos(ee_pos, subgoals, title, save_path=None):
    """
    A function that plots the end effector positions in 3D space.
    """
    x, y, z = ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    #change the background color to white for all three panels
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Plot positions
    ax.scatter(x, y, z, color='blue', label='Positions')
    
    ax.text(x[0]+0.02, y[0], z[0], 'Start', color='black', fontsize=12, fontweight='normal', fontstyle='italic')
    ax.scatter(x[0], y[0], z[0], c='black', marker='o')
    ax.text(x[-1]-0.02, y[-1], z[-1] , 'Goal', color='black', fontsize=12, fontweight='normal', fontstyle='italic')
    ax.scatter(x[-1], y[-1], z[-1], c='black', marker='o')

    # add labels at the end of the segment for each subgoal
    for j,i in enumerate(subgoals[:-1]):
        ax.text(x[i]+0.02, y[i], z[i], f'subgoal {j}', color='black', fontsize=10, fontweight='normal', fontstyle='italic')
        # add a ball marker at the end of the segment
        ax.scatter(x[i], y[i], z[i], c='black', marker='o')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.set_title(title)

    # save the plot
    if save_path is None:
        fig.savefig(f'{title.replace(" ", "-")}.png')
    else:
        fig.savefig(f'{save_path}/{title.replace(" ", "-")}.png')


if __name__ == '__main__':
    config_path = '../../config/kitchen1.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    demo = config["training"]["demo"]
    with h5py.File(config["data"]['data_dir'], 'r') as f: 
        subgoals = f[f"data/demo_{demo}/{config['data']['subgoals_dataset']}"]
        ee_positions = f[f"data/demo_{demo}/abs_actions"][:,:3]
        scatter_ee_pos(ee_positions, subgoals, 'Raw Data: End Effector Positions')

