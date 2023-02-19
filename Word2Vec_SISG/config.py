import numpy as np

class Config():
    def __init__(self):
        self.emb_dim = 300
        self.MIN_COUNT = 5
        self.max_window_size = 5
        self.subsampling_threshold = 1e-04
        self.num_neg_each = 5
        self.num_epoch = 5
        # self.train_file = './data/wiki.en.small.pkl'
        self.starting_lr = 0.025
        self.train_file = './data/wiki.en.pkl'
        self.pickle_path = './pickles'
        self.word_to_index = ''
        self.min_ngram = 3
        self.max_ngram = 6

        # self.emb_save_path = './results/embedding_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
        #                                                               subsampling_t, epochs)
        # self.cont_save_path = './results/context_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
        #                                                              subsampling_t, epochs)
        # self.nodevec_mat_save_path = './results/nodevec_mat_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
        #                                                                   subsampling_t, epochs)