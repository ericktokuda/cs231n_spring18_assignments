#!/usr/bin/env python3
"""Assignments and exercises for CS231n 2016

"""

import os
import numpy as np
import argparse
import pickle

import knn

def load_cifar10(cifardir):
    n = 5
    pref = 'data_batch_'
    data = np.ndarray((50000, 3072)) # 5 partitions of 32x32
    labels = np.ndarray(50000) # 5 partitions of 32x32

    for i in range(n):
        filename = os.path.join(cifardir, pref + str(i+1))
        fh = open(filename, 'rb')
        mydict = pickle.load(fh, encoding='latin1')
        data[i:i+10000, :] = mydict['data']
        labels[i:i+10000] = mydict['labels']
        fh.close()
    return data, labels

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, help='dataset option')
    args = parser.parse_args()

    if args.dataset == 'cifar10':
        cifardir = './data/cifar-10-batches-py/'
        if not os.path.exists(cifardir):
            print('Dowload cifar-10 dataset for python')
            print('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
            return
        X, Y = load_cifar10(cifardir)
    
    classifier = knn.Knn()
    classifier.train(X, Y)
    res = classifier.predict(X[:10, :], 5)

if __name__ == "__main__":
    main()

