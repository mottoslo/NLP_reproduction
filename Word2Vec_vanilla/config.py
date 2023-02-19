import os


class Config:
    def __init__(self):
        self.train_folder_path = \
            './data/clean/'
        # self.train_folder_path = \
            # './data/small_data/'

        # self.train_folder_path = './data/temp2'
        # self.train_folder_path = \
        #     './data/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/'

        # self.train_files = \
            # [os.path.join(self.train_folder_path, filename) for filename in os.listdir(self.train_folder_path)]
        self.evaluation_data = \
            './data/eval_data/semantic_syntactic_testset.txt'
        self.model_type = "CBOW"  #If CBOW, "CBOW" // If Skip-Gram, "SG"
        self.MIN_COUNT = 5
        self.vocabulary_path = './dicts/vocabulary.pkl'
        self.index_to_word_path = './dicts/index_to_word.pkl'
        self.word_to_index_path = './dicts/word_to_index.pkl'
        self.window_size = 10  #maximum window size
        self.huffman_tree_path = './dicts/huffman_tree.pkl'
        self.bin_paths_path = './dicts/bin_paths.pkl'
        self.unigram_table_path = './dicts/unigram_table.pkl'
        self.index_container_path = './dicts/index_container.pkl'
        # self.eval_root_path = "./test_data/questions-words.txt"  # data for evaluation

        self.hidden_size = 300
        self.epoch = 1
        self.loss_type = 'hs'    #'ng' if negative sampling, 'hs' if hierarchical softmax
        self.learning_rate = 0.05
        self.THRESHOLD = 1e-5
        self.NEG = 15            #number of negative samples
        self.subsampling_threshold = 1e-5
        # self.MIN_COUNT = 5


if __name__ == "__main__":
    import pickle
    cfg = Config()
    vocabulary = pickle.load(open(cfg.vocabulary_path, 'rb'))
    print(sum(vocabulary.values()))

