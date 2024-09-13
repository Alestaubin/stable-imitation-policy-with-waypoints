"""
NOTE: This script is taken from Mimicplay's github repository.

Date: Oct 16, 2023

This script addresses relative file path issue by replacing all relative
file path in given hdf5 file with absolute file. This step is necessary 
because Libero library will override relative file path inconsistent with
our project setting.

Example usage:
(recommended): python preprocess_hdf5.py -i ./datasets/playdata/demo.hdf5 -o ./datasets/playdata/demo_modified.hdf5
(if want to update hdf5 file inplace): python preprocess_hdf5.py -i ./datasets/playdata/demo.hdf5 --override
"""

import argparse
import h5py
import os

import xml.etree.ElementTree as ET
import json
from log_config import logger

def fix_path(path, workspace):
    """
    Fix the path by replacing old user's home directory with current

    @param path, input path
    @return absolute path
    """
    if path.find("libero") != -1:
        # this is a path to libero library
        full_path = f"{workspace}/LIBERO/libero/libero/assets/"
    elif path.find("robosuite") != -1:
        # this is a path to robosuite library
        full_path = f"{workspace}/robosuite/robosuite/models/assets/"
    else:
        # idk 
        raise ValueError("Unknown path mannnn!")
    
    rel_path = path.split("assets/")[-1]
    return full_path + rel_path

def replace_relative_path(xml_string, workspace):
    """
    Replacing all relative path in of mesh and texture with absolute path

    @param xml_string, input xml string from model_file fields of hdf5 file
    @return updated xml string with absolute path
    """
    tree = ET.fromstring(xml_string)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures
    
    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        new_path = fix_path(old_path, workspace)
        #new_path = os.path.abspath(old_path)
        elem.set("file", new_path)
    
    return ET.tostring(root, encoding="utf8").decode("utf8")


def process_file(input_path, workspace, libero_folder, output_path=None, bddl_file=None):
    """
    Preprocess given hdf5 file by replacing relative file path with absolute path

    @param input_path, input hdf5 file path
    @param output_path, output path of updated hdf5 file. If no output_path is specified,
                        the function will override input file path inplace
    """
    logger.info(f"Processing file {input_path}")
    # if no output_path, set output_path to input_path, this is equivelent to write inplace
    if not output_path:
        output_path = input_path
    # otherwise, make a full copy of input file, then modify in place
    else:
        os.system(f"cp {input_path} {output_path}")
    
    with h5py.File(output_path, "r+") as f:
        bddl_file_name = f["data"].attrs.get("bddl_file_name")
        #logger.info("file name before: ", bddl_file_name)
        if bddl_file is not None:
            bddl_file_name = f"{workspace}/"+ bddl_file
            logger.info(f"bddl file name is : {bddl_file_name}")
        else:
            bddl_file_name = f"{workspace}/LIBERO/libero/libero/bddl_files/{libero_folder}/" + os.path.basename(bddl_file_name)
        #logger.info("file name after: ", bddl_file_name)
        f["data"].attrs.modify("bddl_file_name", bddl_file_name)
        env_args = f['data'].attrs.get("env_args")
        env_args = json.loads(env_args)
        env_args["bddl_file"] = bddl_file_name
        env_args["env_kwargs"]["bddl_file_name"] = bddl_file_name
        env_args = json.dumps(env_args)
        f['data'].attrs.modify("env_args", env_args)
        for demo_name in f["data"].keys():
            f["data"][demo_name].attrs["model_file"] = replace_relative_path(f["data"][demo_name].attrs["model_file"], workspace)
    
    logger.info(f"File {input_path} processed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Process XML file to replace relative paths.")
    parser.add_argument("-i", "--input", 
                        type=str, 
                        default="./datasets/playdata/demo.hdf5",
                        help="Path to the input XML file")
    parser.add_argument("--override", 
                        action="store_true", 
                        help="Override the input file with changes")
    parser.add_argument("-o", "--output", 
                        type=str, 
                        default=None, 
                        help="Path to the output file (only if not overriding original hdf5 file)")
    parser.add_argument("--workspace",
                        type=str,
                        default="/Users/alexst-aubin/SummerResearch24/V2",
                        help="Path to the workspace directory")
    parser.add_argument("--libero_folder",
                        type=str,
                        default="libero_90",
                        help="name of the libero folder")
    parser.add_argument("--bddl_file",
                        type=str,
                        default=None,
                        help="Path to the bddl file")
    
    args = parser.parse_args()

    if args.override:
        process_file(args.input)
    else:
        if not args.output:
            raise ValueError("You must specify an output file path if not overriding the input file.")
        if os.path.abspath(args.input) == os.path.abspath(args.output):
            raise ValueError("Input and output file paths are the same. Use --override or specify a different output path.")
        process_file(args.input, args.workspace, args.libero_folder, args.output, args.bddl_file)

if __name__ == "__main__":
    main()