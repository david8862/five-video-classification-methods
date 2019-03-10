#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This scipt will split the sequence image dataset
under "data" dir to the target sequence length.
Result will be stored at "data/output". Images
will be rename and corresponding data_file.csv
will be generated.
NOTE: Due to path limit, this script only works
at <Project>/tools dir.
'''

from PIL import Image
import numpy as np
import os, glob, shutil
import csv
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from data import DataSet
from utils.common import touchdir, get_config


def main():
    cf = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_seq_length', help='Sequence length you want to split to', type=int, default=cf.getint('sequence', 'seq_length'))
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

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
                    # assume A~Z + a~z is enough
                    if label == ord('Z') + 1:
                        label = ord('a')
                    count = 0

    with open(os.path.join('data', 'output', 'data_file.csv'), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print('Done. Split data at <Project>/data/output. Pls use it for your train&test')


if __name__ == "__main__":
    main()

