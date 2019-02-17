#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run this scipt under the grayscale dataset path.
it will go through all the subdir and convert
all the grayscle bmp to fakeRGB jpg.Output jpg
file will be at output/ dir.
'''

from PIL import Image
import numpy as np
import os, glob

def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    target_class_path = dst + os.path.sep + class_name
    touchdir(target_class_path)

    output_path = target_class_path + os.path.sep + class_name + '_' + path.split(os.path.sep)[-1].split('.')[-2] + '.jpg'
    rgb_image.save(output_path)
    print(output_path)


def fakeRgb2(path, dst):
    '''
    方法二：最原始的拼接数组方法
    :param path:图片输入路径
    :param dst:图片输出路径
    :return:rgb3个通道值相等的rgb图像
    '''

    b = Image.open(path)
    # 转换为灰度图
    if b.mode != 'L':
        b = b.convert('L')
    # 将图像转为数组
    b_array = np.asarray(b)
    # 将3个二维数组重叠为一个三维数组
    rgb_array = np.zeros((b_array.shape[0], b_array.shape[1], 3), "uint8")
    rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2] = b_array, b_array, b_array
    rgb_image = Image.fromarray(rgb_array)

    class_name = path.split(os.path.sep)[-2]
    target_class_path = dst + os.path.sep + class_name
    touchdir(target_class_path)

    output_path = target_class_path + os.path.sep + class_name + '_' + path.split(os.path.sep)[-1].split('.')[-2] + '.jpg'
    rgb_image.save(output_path)
    print(output_path)


def main():
    class_folders = glob.glob('./*')
    class_folders = [item for item in class_folders if os.path.isdir(item)]
    print(class_folders)
    for class_folder in class_folders:
        img_files = glob.glob(os.path.join(class_folder, '*.bmp'))

        for img_file in img_files:
            fakeRgb1(img_file, 'output')


if __name__ == "__main__":
    main()

