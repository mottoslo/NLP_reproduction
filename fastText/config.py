class Config():
    def __init__(self):
        self.num_epoch = 5
        self.hidden_size = 10
        self.num_classes = {'ag' : 4, 'amzf' : 5, 'amzp' : 2, 'dbp' : 14, 'sogou' : 5, 'yaha' : 10, 'yelpf' : 5, 'yelpp' : 2}
        self.data_type = 'ag'
        self.use_bigram = True
        self.raw_data_folder = './raw_data'
        self.pickle_save_path = './pickles'
        self.MIN_COUNT = 5
        self.starting_lr = 0.1
        self.model_save_path = './result_model'


        self.train_data_save_path = './datasets/train'
        self.test_data_save_path = './datasets/test'