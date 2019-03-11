#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
use TSNE in sklearn to visualize places365 dataset
in 2D or 3D dimension. PCA transform is optional
for decreasing original dimension.
NOTE: Due to path limit, this script only works
at <Project>/tools dir.
'''
from time import time
import os, sys, argparse, random
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from extractor import Extractor


def get_data(sample_num):
    #random pick samples from classes
    random_images = []
    #images at <Projects>/data/places365
    class_folders = glob.glob(os.path.join('data', 'places365', '*'))
    for class_item in class_folders:
        images = glob.glob(os.path.join(class_item, '*.jpg'))
        for _ in range(sample_num):
            # Get a random row.
            sample = random.randint(0, len(images) - 1)
            image = images[sample]
            random_images.append(image)
    random_images = sorted(random_images)

    classes = glob.glob(os.path.join('data', 'places365', '*'))
    classes = sorted([item.split(os.path.sep)[-1] for item in classes])

    # get the feature extract model
    model = Extractor()

    # Now loop through and extract features to build the sequence.
    sequence = []
    labels = []
    pbar = tqdm(total=len(random_images))
    for image in random_images:
        features = model.extract(image)
        sequence.append(features)
        label = image.split(os.path.sep)[-2]
        label = classes.index(label)
        labels.append(label)
        pbar.update(1)

    return np.array(sequence), np.array(labels), len(sequence), len(sequence[0])


def plot_embedding(data, label, title, dim):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    if dim == 2:
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
    elif dim == 3:
        ax = fig.gca(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c = label, s = 20)

        ax.view_init(4, -72)
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsne_dim', help='TSNE dimension to display, 2 or 3. default=2', type=int, default=2)
    parser.add_argument('--pca_level', help='add PCA transform to feature space (optional)', type=float)
    parser.add_argument('--sample_num', help='number of samples to pick from each class, default=500', type=int, default=500)

    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    data, label, n_samples, n_features = get_data(args.sample_num)

    if args.pca_level:
        pca = PCA(n_components=args.pca_level)
        data = pca.fit_transform(data)
        print('feature space dimension change to {} by PCA'.format(pca.n_components_))

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=args.tsne_dim, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    print('t-SNE embedding Done')

    fig = plot_embedding(result, label,
                        't-SNE embedding of the digits (time %.2fs)'
                        % (time() - t0), dim=args.tsne_dim)
    plt.show(fig)


if __name__ == '__main__':
    main()

