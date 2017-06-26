#!/usr/bin/env python

import os, sys, glob, time
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt

def load_file(file_name, is_signal, sub_row, sub_col, threshold, do_plot=False):
    data_zip = np.load(file_name)
    data = data_zip["arr_0"]

    num_rows = data.shape[1] # data.shape[1]
    num_cols = data.shape[2] # data.shape[2]

    # split input data into sub_row x sub_col chunks
    intermediate_arrays = np.split(data[0,:,:], np.arange(sub_row, num_rows, sub_row), axis=0)
    sub_arrays = list()
    for ia in intermediate_arrays:
        sub_arrays += np.split(ia, np.arange(sub_col, num_cols, sub_col), axis=1)

    # only include those chunks that are correct
    images = list()
    for image_ctr, sa in enumerate(sub_arrays):
        if sa.shape != (sub_row, sub_col):
            continue
        if not np.any(sa > threshold):
            continue
        images.append(sa)
        images.append(sa[::-1,:])
        images.append(sa[:,::-1])
        images.append(sa[::-1,::-1])
    input_var = np.asarray(images)

    if is_signal:
        target_var = np.ones(shape=(input_var.shape[0]))
    else:
        target_var = np.zeros(shape=(input_var.shape[0]))

    return input_var, target_var

def concatenate_inputs(file_name, input_matrix, targets, sub_row, sub_col, threshold, is_signal=False):
    temp_input_matrix, temp_targets = load_file(file_name, is_signal, sub_row, sub_col, threshold, do_plot=False)
    if input_matrix is None or targets is None:
        input_matrix = temp_input_matrix
        targets = temp_targets
    else:
        input_matrix = np.concatenate((input_matrix, temp_input_matrix))
        targets = np.concatenate((targets, temp_targets))
    return input_matrix, targets

def load_data(num_bg_file_train, num_bg_file_test, num_sg_file_train, num_sg_file_test, sub_row, sub_col, threshold):
    input_train, input_val, input_test = None, None, None
    target_train, target_val, target_test = None, None, None
    print("background files")
    for bg_ctr, bg_file in enumerate(glob.glob("../rawcam_masked/2016.04.14_co60/*")):
        if bg_ctr < num_bg_file_train:
            print("bg train: ", bg_ctr, bg_file)
            input_train, target_train = concatenate_inputs(bg_file, input_train, target_train, sub_row, sub_col, threshold, is_signal=False)
        elif bg_ctr >= num_bg_file_train and bg_ctr < num_bg_file_test:
            print("bg test: ", bg_ctr, bg_file)
            input_test, target_test = concatenate_inputs(bg_file, input_test, target_test, sub_row, sub_col, threshold, is_signal=False)

    num_bg_train = input_train.shape[0]
    print("background train", num_bg_train)

    num_bg_test = input_test.shape[0]
    print("background test", num_bg_test)

    print("signal files")
    for sg_ctr, sg_file in enumerate(glob.glob("../rawcam_masked/2016.04.14_ra226/*")):
        if sg_ctr < num_sg_file_train:
            print("sg train: ", sg_ctr, sg_file)
            input_train, target_train = concatenate_inputs(sg_file, input_train, target_train, sub_row, sub_col, threshold, is_signal=True)
        elif sg_ctr >= num_sg_file_train and sg_ctr < num_sg_file_test:
            print("sg test: ", sg_ctr, sg_file)
            input_test, target_test = concatenate_inputs(sg_file, input_test, target_test, sub_row, sub_col, threshold, is_signal=True)

    num_sg_train = input_train.shape[0] - num_bg_train
    print("signal train", num_sg_train)

    num_sg_test = input_test.shape[0] - num_bg_test
    print("signal test", num_sg_test)

    input_train = input_train.astype(np.float32)
    target_train = target_train.astype(np.int32)

    input_test = input_test.astype(np.float32)
    target_test = target_test.astype(np.int32)

    return (input_train, target_train), (input_test, target_test)
