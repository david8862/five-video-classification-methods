#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
use TSNE in sklearn to visualize dataset in 2D
or 3D dimension. PCA transform is optional for
decreasing original dimension.
NOTE: Due to path limit, this script only works
at <Project>/tools dir.
'''
from time import time
import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from data import DataSet
from utils.common import get_config


def expand_label(label, seq_length):
    new_label = []
    for i in range(label.shape[0]):
        for j in range(seq_length):
            new_label.append(np.argmax(label[i]))

    return np.array(new_label)

def get_data(data_type):
    cf = get_config()
    seq_length = cf.getint('sequence', 'seq_length')
    data = DataSet(seq_length=seq_length, class_limit=None)
    X_train, y_train = data.get_all_sequences_in_memory(data_type, data_type='features')
    X_train = X_train.reshape(-1,X_train.shape[-1])

    n_samples, n_features = X_train.shape
    y_train = expand_label(y_train, seq_length=seq_length)
    return X_train, y_train, n_samples, n_features


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
    parser.add_argument('--data_type', help='train or test data, default=train', type=str, default='train')

    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    data, label, n_samples, n_features = get_data(args.data_type)

    if args.pca_level:
        pca = PCA(n_components=args.pca_level)
        data = pca.fit_transform(data)
        print('feature space dimension change to {} by PCA'.format(pca.n_components_))

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=args.tsne_dim, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)

    fig = plot_embedding(result, label,
                        't-SNE embedding of the digits (time %.2fs)'
                        % (time() - t0), dim=args.tsne_dim)
    plt.show(fig)


if __name__ == '__main__':
    main()

