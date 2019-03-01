#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
this scipt will split the sequence image dataset
under "data" dir to the target sequence length.
Images will be rename and data_file.csv will be
updated
'''

from PIL import Image
import numpy as np
import os, glob, shutil
import csv
import argparse
from data import DataSet

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_seq_length', help='Sequence length you want to split to', type=int, default=5)
    args = parser.parse_args()

    # Get the dataset.
    data = DataSet.get_data()
    data_file = []

    for sample in data:
        # check current target_seq_length
        if int(sample[3]) >= args.target_seq_length:
            images = DataSet.get_frames_for_sample(sample)
            label = ord('A')
            count = 0
            for image in images:
                path_suffix = image.split('-')[-1]
                # move image to "data/output" dir to avoid file name chaos
                target_path = os.path.join('data', 'output', sample[0], sample[1])
                touchdir(target_path)
                path_prefix = os.path.join(target_path, sample[2]) + str(chr(label)) + '-'
                # move the image
                os.rename(image, path_prefix + path_suffix)
                count = count + 1
                if(count == args.target_seq_length):
                    data_file.append([sample[0], sample[1], sample[2] + str(chr(label)), count])
                    label = label + 1
                    count = 0

    with open(os.path.join('data', 'output', 'data_file.csv'), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print('Done')


if __name__ == "__main__":
    main()

