import numpy as np
import matplotlib.pyplot as plt
from lib.utils.log_config import logger

def scatter_waypoints(waypoint_position, waypoint_velocity, title):
    """
    A function that plots the waypoints and their velocities in 3D space.
    """
    x, y, z = waypoint_position[:, 0], waypoint_position[:, 1], waypoint_position[:, 2]
    u, v, w = 0.1*waypoint_velocity[:, 0], 0.1*waypoint_velocity[:, 1], 0.1*waypoint_velocity[:, 2]

    # Create a 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot positions
    ax.scatter(x, y, z, color='blue', label='Positions')

    # Plot velocity vectors (arrows)
    ax.quiver(x, y, z, u, v, w, color='red', label='Velocities')

    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    ax.set_title(title)

    # save the plot
    fig.savefig(f'plots/waypoints/{title.replace(" ", "-")}.png')

def plot_rollouts(data, policies, path):
    for i, ds_policy in enumerate(policies):
        waypoint_position = data["subgoal_" + str(i)]["waypoint_position"]
        if waypoint_position.shape[1] == 3:
            dt = 0.01
            start_point = waypoint_position[0].reshape(1, waypoint_position.shape[1])
            goal_point = waypoint_position[-1].reshape(1, waypoint_position.shape[1])

            rollout = [start_point]
            distance_to_target = np.linalg.norm(rollout[-1] - goal_point)
            while distance_to_target > 0.01 and len(rollout) < 5e3:
                vel = ds_policy.predict(rollout[-1])

                if not isinstance(dt, np.ndarray):
                    dt = np.array(dt, dtype=np.float32)
                if not isinstance(vel, np.ndarray):
                    vel = np.array(vel, dtype=np.float32)

                rollout.append(rollout[-1] + dt * vel)
                distance_to_target = np.linalg.norm(rollout[-1] - goal_point)

            rollout = np.array(rollout).squeeze()
            print(f'Rollout finished with distance to target: {distance_to_target}')

            x, y, z = waypoint_position[:, 0], waypoint_position[:, 1], waypoint_position[:, 2]
            x_rollout, y_rollout, z_rollout = rollout[:, 0], rollout[:, 1], rollout[:, 2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x, y, z, c='b', label='waypoints')
            ax.plot(x_rollout, y_rollout, z_rollout, c='r', label='rollout')
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')
            ax.legend()
            ax.set_title('DS Policy Waypoints')
            fig.savefig(f'{path}/3D-waypoints-policy-subgoal-{i}.png')

            fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

            # X-Y projection
            axes[0].plot(x, y, 'b', label='waypoints')
            axes[0].plot(x_rollout, y_rollout, 'r', label='rollout')
            axes[0].set_xlabel('X1')
            axes[0].set_ylabel('X2')
            axes[0].set_title('X-Y Projection')
            axes[0].legend()

            # X-Z projection
            axes[1].plot(x, z, 'b', label='waypoints')
            axes[1].plot(x_rollout, z_rollout, 'r', label='rollout')
            axes[1].set_xlabel('X1')
            axes[1].set_ylabel('X3')
            axes[1].set_title('X-Z Projection')
            axes[1].legend()

            # Y-Z projection
            axes[2].plot(y, z, 'b', label='waypoints')
            axes[2].plot(y_rollout, z_rollout, 'r', label='rollout')
            axes[2].set_xlabel('X2')
            axes[2].set_ylabel('X3')
            axes[2].set_title('Y-Z Projection')
            axes[2].legend()

            plt.suptitle('2D Projections of the 3D Plot')

            fig2.savefig(f'{path}/2D-waypoints-policy-subgoal-{i}.png')

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
