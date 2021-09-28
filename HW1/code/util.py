import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from os.path import join

def get_num_CPU():
    '''
    Counts the number of CPUs available in the machine.
    '''
    return multiprocessing.cpu_count()

def display_filter_responses(opts, response_maps):
    '''
    Visualizes the filter response maps.

    [input]
    * response_maps: a numpy.ndarray of shape (H,W,3F)
    '''
    
    n_scale = len(opts.filter_scales)
    plt.figure(1)
    
    for i in range(n_scale*4):
        plt.subplot(n_scale, 4, i+1)
        resp = response_maps[:, :, i*3:i*3 + 3]
        resp_min = resp.min(axis=(0,1), keepdims=True)
        resp_max = resp.max(axis=(0,1), keepdims=True)
        resp = (resp - resp_min)/(resp_max - resp_min)
        plt.imshow(resp)
        plt.axis("off")

    plt.subplots_adjust(left=0.05,right=0.95,top=0.95,bottom=0.05,wspace=0.05,hspace=0.05)
    plt.savefig(join(opts.result_dir, 'responses.jpg'))
    plt.show()

def visualize_wordmap(wordmap, out_path=None):
    plt.figure(2)
    plt.axis('equal')
    plt.axis('off')
    plt.imshow(wordmap)
    plt.show()
    if out_path:
        plt.savefig(out_path, pad_inches=0)
