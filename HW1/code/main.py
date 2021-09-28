from os.path import join
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
from opts import get_opts

import pickle, csv

def main():
    opts = get_opts()
    
    # '''====================Hyperparameters Tuning===================='''
    # sigmas_lst = [[1,2,4,8], [1,4,8,16], [1,2,4,8,8*np.sqrt(2)]]
    # Ls = np.arange(4, 2, -1)
    # Ks = np.array([100, 130, 160, 190, 220, 250, 300])
    # alphas = np.array([180, 210, 240, 270, 300, 350, 400])
    # res = []  # store parameters and results

    # print('='*20 + "Start Tuning" + '='*20)
    # print()
    # with open('res.csv', 'w') as f1:
    #     writer = csv.writer(f1, delimiter='\t', lineterminator='\n')
    #     for sigmas in sigmas_lst:
    #         for L in Ls:
    #             for K in Ks:
    #                 for alpha in alphas:
    #                     opts.filter_scales = sigmas
    #                     opts.L = L
    #                     opts.K = K
    #                     opts.alpha = alpha

    #                     print('Parameters: K=', opts.K, ' alpha=', opts.alpha, ' L=', opts.L, ' sigmas=', opts.filter_scales)
    #                     n_cpu = util.get_num_CPU()
    #                     visual_words.compute_dictionary(opts, n_worker=n_cpu)
    #                     visual_recog.build_recognition_system(opts, n_worker=n_cpu)
    #                     conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    #                     writer.writerow([L, K, alpha, conf, accuracy])
    #                     res.append([L, K, alpha, conf, accuracy])
    #                     print('Accuracy:', res[-1][4])
    #                     print('='*50)
    #                     print()
    
    # with open('res.pickle', 'wb') as f:
    #     pickle.dump(res, f)
    # '''====================Hyperparameters Tuning===================='''

    # Q1.1
    img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)
    
    # Q1.3
    img_path = join(opts.data_dir, 'kitchen/sun_abujclohwuaugvev.jpg')
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap, join(opts.result_dir, 'visual_words.jpg'))

    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy = visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    
    print(conf)
    print(accuracy)
    np.savetxt(join(opts.out_dir, 'confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
