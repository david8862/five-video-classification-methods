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



def fakeRGB_convert(path, dst):
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

    class_name = path.split(os.path.sep)[-2]
    # Touch output class path
    output_class_path = os.path.join(dst, class_name)
    touchdir(output_class_path)

    # Store fakeRGB image to output path
    output_full_path = os.path.join(dst, path)
    rgb_image.save(output_full_path)
    print(output_full_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='Output path for the converted image', type=str, default=os.path.join(os.path.dirname(__file__), 'output'))
    args = parser.parse_args()

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
            fakeRGB_convert(img_file, args.output_path)



if __name__ == "__main__":
    main()

