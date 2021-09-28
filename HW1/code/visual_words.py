import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
import sklearn.cluster
from scipy.spatial import distance

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts   : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    
    # Check range and re-normalize
    if np.max(img) > 1:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # Check number of channels
    if len(img.shape) == 2:
        img = np.stack((img,)*3, axis=2)
    
    img = skimage.color.rgb2lab(img)
    responses = None  # image collage of 3 x #S x #F filter responses
    for s in filter_scales:
        # Gaussian filter on three channels
        for ch in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,ch], s, order=0)
            if responses is None:
                responses = res[:,:,np.newaxis]
            else:
                responses = np.concatenate((responses, res[:,:,np.newaxis]), axis=2)
        for ch in range(3):
            res = scipy.ndimage.gaussian_laplace(img[:,:,ch], s)
            responses = np.concatenate((responses, res[:,:,np.newaxis]), axis=2)
        for ch in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,ch], s, order=(0,1))
            responses = np.concatenate((responses, res[:,:,np.newaxis]), axis=2)
        for ch in range(3):
            res = scipy.ndimage.gaussian_filter(img[:,:,ch], s, order=(1,0))
            responses = np.concatenate((responses, res[:,:,np.newaxis]), axis=2)

    return responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    [input]
    * args[0]   : the index of the training image in train_files
    * args[1]   : opts passed to compute_dictionary
    * args[2]   : path to a training image
    '''

    opts = args[1]
    alpha = opts.alpha  # number of random pixels per image
    img_path = join(opts.data_dir, args[2])
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    h, w = img.shape[:2]
    filter_responses = extract_filter_responses(opts, img)
    filter_responses = filter_responses.reshape(h * w, -1)
    rand_indices = np.random.choice(h * w, alpha, replace=False)
    filter_responses = filter_responses[rand_indices]
    np.save(join(opts.temp_dir, 'response_%s.npy' % args[0]), filter_responses)
    
    return


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    temp_dir = opts.temp_dir
    K = opts.K
    args_lst = []   # a list of arguments passed to the function on each process
    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    
    for i in range(len(train_files)):
        args_lst.append([i, opts, train_files[i]])        

    # Use data based parallelism, i.e. Pool
    p = multiprocessing.Pool(processes=n_worker)
    p.map(compute_dictionary_one_image, args_lst)
    p.close()
    p.join()
    
    # Read responses from temp_dir
    responses = np.load(join(temp_dir, 'response_0.npy'))
    for i in range(1, len(train_files)):
        resp = np.load(join(temp_dir, 'response_%s.npy' % i))
        responses = np.vstack((responses, resp))
    
    # Run KMeans
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(responses)
    dictionary = kmeans.cluster_centers_
    
    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    print("Dictionary shape [K x 3SF]: ", dictionary.shape)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts   : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    img_responses = extract_filter_responses(opts, img)
    distances = distance.cdist(dictionary, img_responses.reshape(-1, img_responses.shape[2]))  # shape K x H*W
    clusters = np.argmin(distances, axis=0)  # Find min index (the cluster each pixel belongs to) per column

    return clusters.reshape(img.shape[0], -1)  # Each min index corresponds to one cluster from 0 to K-1