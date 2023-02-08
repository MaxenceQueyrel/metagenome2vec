# -*- coding: utf-8 -*-

import subprocess
import os

from metagenome2vec.utils.string_names import *


def create_dir(path, mode, sub_path=None):
    """
    Create a dir if not exist

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :param sub_path: String, slash separated string that compeltes path
    :return:
    """
    if mode == "local" or mode == "s3":
        if sub_path is not None:
            path = os.path.join(path, sub_path)
        if not os.path.exists(path):
            os.makedirs(path)
    if mode == "hdfs":
        if sub_path is None:
            if 0 == dir_exists(path, mode):
                subprocess.call("hdfs dfs -mkdir %s" % path, shell=True)
        else:
            if sub_path[-1] == "/":
                sub_path = sub_path[:-1]
            path_curr = path
            for folder in sub_path.split("/"):
                if 0 == dir_exists(os.path.join(path_curr, folder), mode):
                    subprocess.call(
                        "hdfs dfs -mkdir %s" % (os.path.join(path_curr, folder)),
                        shell=True,
                    )
                path_curr = os.path.join(path_curr, folder)


def remove_dir(path, mode):
    """
    Remove a directory

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if dir_exists(path, mode):
        if mode == "s3":
            cmd_rm = ["aws", "s3", "rm", "--recursive"]
        elif mode == "local":
            cmd_rm = ["rm", "-r"]
        else:
            cmd_rm = ["hdfs", "dfs", "-rm", "-r"]
        subprocess.call(cmd_rm + [path])


def move_file(path, mode):
    """
    Move a file

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if mode == "s3":
        cmd_mv = ["aws", "s3", "mv", "--recursive"]
    elif mode == "local":
        cmd_mv = ["mv"]
    else:
        cmd_mv = ["hdfs", "dfs", "-mv"]
    subprocess.call(cmd_mv + [path])


def copy_dir(path, mode):
    """
    Copy a directory

    :param path: String, the dir we want to create if not exists
    :param mode: String : hdfs, local or s3
    :return:
    """
    if mode == "s3":
        cmd_cp = ["aws", "s3", "cp", "--recursive"]
    elif mode == "local":
        cmd_cp = ["cp", "-r"]
    else:
        cmd_cp = ["hdfs", "dfs", "-cp"]
    subprocess.call(cmd_cp + [path])


def dir_exists(path, mode):
    """
    Tell if a dir exists or not

    :param path: String
    :param mode: String : hdfs, local or s3
    :return: 1 if exists else 0
    """
    if mode == "hdfs":
        return int(
            subprocess.check_output(
                "hdfs dfs -test -d %s && echo 1 || echo 0" % path, shell=True
            )
        )
    if mode == "local":
        return 1 if os.path.isdir(path) else 0
    if mode == "s3":
        try:
            res = (
                subprocess.check_output("aws s3 ls %s" % path, shell=True)
                .split(" ")[-1]
                .replace("\n", "")
                .replace("/", "")
            )
            return int(os.path.basename(path) == res)
        except Exception:
            return 0


def generate_list_file(path_data, mode, sub_folder=True):
    """
    Generate a list of file name which come from the path_data.
    This function follows the architecture of biobank.
    :param path_data: String, the path where are stored the data on hdfs
    :param mode: String : hdfs, local or s3
    :param sub_folder: True, This argument is True when we want to get a list of sub list
    :return:
    l_res : List, list like [sub_folder_name_1, sub_folder_name_2]
    or
    l_res : List, list like [sub_folder_name_1, [path_file_1, path_file_2, path_file_3...],
                                   sub_folder_name_2, ...]
    """
    if mode == "hdfs":
        # create_dir(os.path.join(os.path.dirname(root_folder), "data"), "local")
        # path_subfolder = os.path.join(os.path.dirname(root_folder), "data/list_path_hdfs/%s_subfolder_%s.pkl" % (path_data.split("/")[-1], sub_folder))
        # if os.path.isfile(path_subfolder):
        #    with open(path_subfolder) as f:
        #        return pickle.load(f)
        # Get the list of files / folders in path_data_hdfs
        list_file_hdfs = subprocess.check_output(
            "hdfs dfs -ls %s" % path_data, shell=True
        )
        list_file_hdfs = (
            list_file_hdfs.decode("utf-8")
            if isinstance(list_file_hdfs, bytes)
            else list_file_hdfs
        )
        list_file_hdfs = map(
            lambda x: x.split(" ")[-1], list_file_hdfs.split("\n")[1:]
        )[:-1]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            # with open(path_subfolder, 'w') as f:
            #     pickle.dump(list_file_hdfs, f)
            return list_file_hdfs
        l_res = []
        for file_hdfs in list_file_hdfs:
            list_file_hdfs_2 = subprocess.check_output(
                "hdfs dfs -ls %s" % file_hdfs, shell=True
            )
            list_file_hdfs_2 = map(
                lambda x: x.split(" ")[-1], list_file_hdfs_2.split("\n")[1:]
            )[:-1]
            list_file_hdfs_2 = (
                list_file_hdfs_2.decode("utf-8")
                if isinstance(list_file_hdfs_2, bytes)
                else list_file_hdfs_2
            )
            l_res.append(
                [
                    os.path.basename(file_hdfs),
                    [file_hdfs_2 for file_hdfs_2 in list_file_hdfs_2],
                ]
            )
        # with open(path_subfolder, 'w') as f:
        #     pickle.dump(l_res, f)
        return l_res
    if mode == "s3":
        # Get the list of files / folders in path_data_s3
        list_file_s3 = subprocess.check_output("aws s3 ls %s" % path_data, shell=True)
        list_file_s3 = (
            list_file_s3.decode("utf-8")
            if isinstance(list_file_s3, bytes)
            else list_file_s3
        )
        list_file_s3 = [
            x.split(" ")[-1].replace("/", "") for x in list_file_s3.split("\n")[:-2]
        ]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            return list_file_s3
        l_res = []
        for file_s3 in list_file_s3:
            list_file_s3_2 = subprocess.check_output(
                "aws s3 ls %s" % os.path.join(path_data, file_s3) + "/", shell=True
            )
            list_file_s3_2 = (
                list_file_s3_2.decode("utf-8")
                if isinstance(list_file_s3_2, bytes)
                else list_file_s3_2
            )
            list_file_s3_2 = [
                x.split(" ")[-1].replace("/", "")
                for x in list_file_s3_2.split("\n")[:-1]
            ]
            l_res.append(
                [
                    file_s3,
                    [
                        os.path.join(path_data, file_s3, file_s3_2)
                        for file_s3_2 in list_file_s3_2
                    ],
                ]
            )
        return l_res
    if mode == "local":
        # Get the list of files / folders in path_data_hdfs
        list_file = subprocess.check_output("ls %s" % path_data, shell=True)
        list_file = (
            list_file.decode("utf-8") if isinstance(list_file, bytes) else list_file
        )
        list_file = [x.split(" ")[-1] for x in list_file.split("\n")[:-1]]
        # Create a list with the name of the folder we want to del to get only what we want
        if not sub_folder:
            return [os.path.join(path_data, x) for x in list_file]
        l_res = []
        for file in list_file:
            list_file_2 = subprocess.check_output(
                "ls %s" % os.path.join(path_data, file), shell=True
            )
            list_file_2 = (
                list_file_2.decode("utf-8")
                if isinstance(list_file_2, bytes)
                else list_file_2
            )
            list_file_2 = [x.split(" ")[-1] for x in list_file_2.split("\n")[:-1]]
            l_res.append(
                [
                    os.path.basename(file),
                    [
                        os.path.join(path_data, file, file_hdfs_2)
                        for file_hdfs_2 in list_file_2
                    ],
                ]
            )
        return l_res
