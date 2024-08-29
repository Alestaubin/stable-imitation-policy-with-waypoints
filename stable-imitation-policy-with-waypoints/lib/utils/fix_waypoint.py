import h5py
import numpy as np

with h5py.File("data/KITCHEN_SCENE1_put_the_black_bowl_on_the_plate/image_demo_local_with_AWE_waypoints.hdf5", 'a') as f: 
    waypoints = f[f"data/demo_{1}/AWE_waypoints_dp_err005"]
    del f[f"data/demo_{1}/AWE_waypoints_dp_err005"]
    waypoints = np.array(waypoints)
    print (waypoints)
    waypoints[8] = 53
    print (waypoints)
    f.create_dataset(f"data/demo_{1}/AWE_waypoints_dp_err005", data=waypoints)


