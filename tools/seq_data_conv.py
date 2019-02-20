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


data_list = []

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def add_to_datalist(file_prefix):
    global data_list
    if file_prefix not in data_list:
        data_list.append(file_prefix)

def create_datafile():
    global data_list
    data_file = []
    for data_path in data_list:
        nb_frames = os.popen('ls -l ' + data_path + '-* | wc -l').read()
        nb_frames = int(nb_frames)

        parts = data_path.split(os.path.sep)
        filename_no_ext = parts[2]
        classname = parts[1]
        train_or_test = parts[0]

        data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)


def fakeRgb1(path, dst):
    '''
    方法1：直接使用convert将L转为RGB
    :param path:图片输出路径
    :param dst:图片输出路径
    :return:rgb3个通道值相等的rgb图像
    '''
    b = Image.open(path)
    # 转换为灰度图
    if b.mode != 'L':
        b = b.convert('L')
    b = b.convert('RGB')
    # 将图像转为数组
    rgb_array = np.asarray(b)
    # 将数组转换为图像
    rgb_image = Image.fromarray(rgb_array)

    class_name = path.split(os.path.sep)[-2]

    file_name = path.split(os.path.sep)[-1]
    file_name_no_ext = file_name.split('.')[-2]
    file_name_list = file_name_no_ext.split('_')

    target_class_path = dst + os.path.sep + class_name
    touchdir(target_class_path)

    target_file_prefix = class_name + '_' + file_name_list[0]
    target_file_name = target_file_prefix + '-' + file_name_list[1] + '.jpg'

    output_path = target_class_path + os.path.sep + target_file_name
    rgb_image.save(output_path)
    print(output_path)

    add_to_datalist(target_class_path + os.path.sep + target_file_prefix)


#def fakeRgb2(path, dst):
    #'''
    #方法二：最原始的拼接数组方法
    #:param path:图片输入路径
    #:param dst:图片输出路径
    #:return:rgb3个通道值相等的rgb图像
    #'''

    #b = Image.open(path)
    ## 转换为灰度图
    #if b.mode != 'L':
        #b = b.convert('L')
    ## 将图像转为数组
    #b_array = np.asarray(b)
    ## 将3个二维数组重叠为一个三维数组
    #rgb_array = np.zeros((b_array.shape[0], b_array.shape[1], 3), "uint8")
    #rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2] = b_array, b_array, b_array
    #rgb_image = Image.fromarray(rgb_array)

    #class_name = path.split(os.path.sep)[-2]

    #file_name = path.split(os.path.sep)[-1]
    #file_name_no_ext = file_name.split('.')[-2]
    #file_name_list = file_name_no_ext.split('_')

    #target_class_path = dst + os.path.sep + class_name
    #touchdir(target_class_path)

    #target_file_prefix = class_name + '_' + file_name_list[0]
    #target_file_name = target_file_prefix + '-' + file_name_list[1] + '.jpg'

    #output_path = target_class_path + os.path.sep + target_file_name
    #rgb_image.save(output_path)
    #print(output_path)

    #add_to_datalist(target_class_path + os.path.sep + target_file_prefix)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='Output path for the converted image', type=str, default=os.path.join(os.path.dirname(__file__), 'train'))
    args = parser.parse_args()

    class_folders = glob.glob('./*')
    class_folders = [item for item in class_folders if os.path.isdir(item)]
    for class_folder in class_folders:
        jpg_files = glob.glob(os.path.join(class_folder, '*.jpg'))
        bmp_files = glob.glob(os.path.join(class_folder, '*.bmp'))
        img_files = jpg_files + bmp_files

        for img_file in img_files:
            fakeRgb1(img_file, args.output_path)

    create_datafile()


if __name__ == "__main__":
    main()

