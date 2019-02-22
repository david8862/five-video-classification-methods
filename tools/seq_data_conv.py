#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt under the grayscale dataset path.
it will go through all the subdir and convert
all the grayscle bmp/jpg to fakeRGB jpg and
generate data_file.csv record. Output jpg
file will be at specified target dir.
'''

from PIL import Image
import numpy as np
import os, glob
import csv
import argparse


def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_datafile(sequence_path_list):
    data_file = []
    # Loop for the sequence list
    for sequence_path in sequence_path_list:
        # get frame number for the sequence
        nb_frames = os.popen('ls -l ' + sequence_path + '-* | wc -l').read()
        nb_frames = int(nb_frames)

        # Parse sequence info from full path
        # A sequence path would be like:
        #
        # train/bedroom/bedRoom_A97
        #    ^     ^       ^
        #    |     |       |
        # [type][class][seq name]
        #
        parts = sequence_path.split(os.path.sep)
        filename_no_ext = parts[2]
        classname = parts[1]
        train_or_test = parts[0]

        data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

    # save to data_file.csv
    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def fakeRGB_seq_convert(path, dst):
    '''
    Solution1：use PIL image.convert()
    :param path: input image full path
    :param dst: output store folder path
    :return: sequence prefix with path
    '''
    b = Image.open(path)
    # Convert to gray first if is not
    if b.mode != 'L':
        b = b.convert('L')
    # Do fakeRGB
    b = b.convert('RGB')
    # to numpy array
    rgb_array = np.asarray(b)
    # back to RGB image
    rgb_image = Image.fromarray(rgb_array)

    # Parse data sample image info from full path
    # A data sample path would be like:
    #
    # bedroom/A97_000900.jpg
    #    ^     ^      ^
    #    |     |      |
    #[class][batch][unique number]
    #
    class_name = path.split(os.path.sep)[-2]

    file_name = path.split(os.path.sep)[-1]
    file_name_no_ext = file_name.split('.')[-2]
    file_name_parts = file_name_no_ext.split('_')

    # Touch output class path
    output_class_path = os.path.join(dst, class_name)
    touchdir(output_class_path)

    # Combine output file name and sequence prefix
    output_seq_prefix = class_name + '_' + file_name_parts[0]
    output_file_name = output_seq_prefix + '-' + file_name_parts[1] + '.jpg'

    # Store fakeRGB image to output path
    output_full_path = os.path.join(output_class_path, output_file_name)
    rgb_image.save(output_full_path)
    print(output_full_path)

    # Return sequence full prefix. Will be used for data file generation
    return os.path.join(output_class_path, output_seq_prefix)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='Output path for the converted image', type=str, default=os.path.join(os.path.dirname(__file__), 'train'))
    args = parser.parse_args()

    sequence_path_list = []

    # Scan "./" to get class folder
    class_folders = glob.glob('./*')
    class_folders = [item for item in class_folders if os.path.isdir(item)]
    for class_folder in class_folders:
        # Get the image file list. Now handle both jpg and bmp file
        jpg_files = glob.glob(os.path.join(class_folder, '*.jpg'))
        bmp_files = glob.glob(os.path.join(class_folder, '*.bmp'))
        img_files = jpg_files + bmp_files

        # Do the fake RGB convert to every image, converted
        # output will store under output_path with corresponding
        # class sub folder
        for img_file in img_files:
            output_sequence = fakeRGB_seq_convert(img_file, args.output_path)

            if output_sequence not in sequence_path_list:
                sequence_path_list.append(output_sequence)

    # create data_file.csv
    create_datafile(sequence_path_list)


if __name__ == "__main__":
    main()

