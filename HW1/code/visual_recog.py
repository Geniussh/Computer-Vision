import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    hist, bin = np.histogram(wordmap, bins=np.arange(K + 1), density=True)
    # Remember to use bins=K+1 because all but the last bin is half-open, [K, K+1]

    return hist

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.
    Dynamic Programming.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K  # number of visual words
    L = int(opts.L)  # total number of layers is L + 1
    H, W = wordmap.shape[:2]
    num_cells = (4**(L+1) - 1) // 3  # each cell contains a 1xK histogram
    cells = [None] * num_cells  # store the histograms

    # Compute histograms of the finest layer
    for i, y in enumerate(np.array_split(np.arange(H), 2**L)):
        for j, x in enumerate(np.array_split(np.arange(W), 2**L)):
            hist, _ = np.histogram(wordmap[y[0]:y[-1], x[0]:x[-1]], 
                                    bins=np.arange(K + 1), density=True)
            cells[(4**L - 1) // 3 + (i * (2**L) + j)] = hist
            # because the indices in "cells" for the last layer are
            # starting from 4^0 + 4^1 + ... + 4^(L-1)

    # Compute histograms of coarser layers
    for l in range(L-1, -1, -1):
        index = (4**l - 1) // 3   # index to start aggregating histograms for this layer
        
        # Iterating over i, the index of the cells in the finer layer
        # and over j, the offset in each row of the finer layer 
        for i in range((4**(l+1) - 1) // 3, (4**(l+2) - 1) // 3, 2**(l+2)):
            for j in range(0, 2**(l+1), 2):
                hist = cells[i + j] + cells[i + j + 2**(l+1)] + \
                    cells[i + j + 1] + cells[i + j + 1 + 2**(l+1)]
                cells[index] = hist / np.sum(hist)  # L1 normalize
                index += 1
    
    # Apply weights
    cells = np.array(cells)
    weights = np.zeros((num_cells, K))
    for l in range(L + 1):
        weights[(4**l - 1) // 3:(4**(l+1) - 1) // 3] = 2**(l-L-1) if l != 0 else 1/4
    cells = (cells * weights).reshape(1, -1) # flatten to a vector with dim=K*(4^(L+1)-1)/3
    cells = cells / np.sum(cells)  # L1 normalize

    return cells.ravel()

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    img_path = join(opts.data_dir, img_path)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)

    return feature


def get_image_feature_caller(args):
    '''
    Caller to get_image_feature with arguments passed to each subprocess
    '''
    i, opts, img_path, dictionary = args
    feature = get_image_feature(opts, img_path, dictionary)
    # print("Image %s: Features Extracted" % i)
    return feature


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    args = []
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    for i in range(len(train_files)):
        args.append([i, opts, train_files[i], dictionary])

    # Use data based parallelism, i.e. Pool
    p = multiprocessing.Pool(processes=n_worker)
    features = p.map(get_image_feature_caller, args)
    p.close()
    p.join()
    features = np.array(features)
    
    # Save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    print("Features shape [T x K*(4^(L+1)-1)/3]: ", features.shape)


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''
    
    hist_dist = 1 - np.sum(np.minimum(word_hist, histograms), axis=1)
    return hist_dist  # smaller distance means nearer
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = int(trained_system['SPM_layer_num'])
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    train_labels = trained_system['labels']
    args = []

    num_class = len(np.unique(trained_system['labels']))  # Number of classes
    
    for i in range(len(test_files)):
        args.append([i, test_opts, test_files[i], dictionary, 
                        trained_system['features'], train_labels])
    
    p = multiprocessing.Pool(processes=n_worker)
    preds = p.map(evaluate, args)
    p.close()
    p.join()

    # Confusion matrix and accuracy
    preds = np.array(preds)
    conf = np.zeros((num_class, )*2)
    # Size Check
    if len(preds) != len(test_labels):
        print("Size Dismatch btw preds and labels")
    for i in range(len(preds)):
        conf[test_labels[i]][preds[i]] += 1
    
    accuracy = np.trace(conf) / np.sum(conf)

    return conf, accuracy

def evaluate(args):
    '''
    Wrapper to evaluate a test image.

    [input]
    * args     : args passed to each subprocess

    [output]
    * label    : predicted label
    '''
    i, opts, img_path, dictionary, features, train_labels = args
    feature = get_image_feature(opts, img_path, dictionary)
    # print("Test Image %s: Features Extracted" % i)
    
    hist_dist = distance_to_set(feature, features)
    return train_labels[np.argmin(hist_dist)]