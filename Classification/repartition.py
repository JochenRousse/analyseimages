#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from shutil import move
import numpy as np
from distutils.dir_util import copy_tree


def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)


def repartition(path_to_data, path_to_test_data, train_ratio):
    # get dirs
    _, dirs, _ = next(os.walk(path_to_data))

    # calculates how many train data per class
    data_counter_per_class = np.zeros((len(dirs)))
    for i in range(len(dirs)):
        path = os.path.join(path_to_data, dirs[i])
        files = get_files_from_folder(path)
        data_counter_per_class[i] = len(files)
    test_counter = np.round(data_counter_per_class * (1 - train_ratio))

    # transfers files
    for i in range(len(dirs)):
        path_to_original = os.path.join(path_to_data, dirs[i])

        files = get_files_from_folder(path_to_original)
        # moves data
        for j in range(int(test_counter[i])):
            dst = os.path.join(path_to_test_data, files[j])
            src = os.path.join(path_to_original, files[j])
            move(src, dst)


if __name__ == "__main__":
    DB_src = './data/src/'
    DB_train = './data/train/'
    DB_test = './data/test/'

    if not os.path.exists(DB_train):
        os.mkdir(DB_train)
    if not os.path.exists(DB_test):
        os.mkdir(DB_test)

    copy_tree(DB_src, DB_train)

    repartition(DB_train, DB_test, 0.7)
