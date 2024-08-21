import argparse
import h5py

def main(args):
    with h5py.File(args.dataset, 'a') as f:
        demos = list(f["data"].keys())
        print("demos: ", demos)
        for demo in demos:
            for key in list(f[f"data/{demo}"].keys()):
                if key not in args.list:
                    print(f"Deleting data/{demo}/{key}")
                    del f[f"data/{demo}/{key}"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="robomimic/datasets/lift/ph/low_dim.hdf5",
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        '-l',
        '--list', 
        nargs='+', 
        default=["abs_actions", "abs_obs", "obs", "actions", "rewards", "dones", "initial_state", "states"],
        help='<Required> Set flag' 
        )

    main(parser.parse_args())
