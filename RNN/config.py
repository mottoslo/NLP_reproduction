import os

class Config:
    def __init__(self):
        # self.train_folder_path = \
            # './data/clean/'
        self.data_folder_path = './data/'
        self.en_vocab_path = './data/en_vocab'
        self.de_vocab_path = './data/de_vocab'
        self.en_train_path = './data/en_train'
        self.de_train_path = './data/de_train'
        self.en_val_path = './data/en_val'
        self.de_val_path = './data/de_val'
        self.en_test_path = './data/en_test2015'
        self.de_test_path = './data/de_test2015'
        self.model_save_path = f'./results'
        self.max_seq_len = 51
        self.max_vocab_size = 50001
        self.lstm_hidden_size = 1000
        self.emb_dim = 1000
        self.lstm_n_layers = 4
        self.lstm_drop_prob = 0.2
        self.batch_size = 128
        self.num_epochs = 10
        self.decoder_output_size = 123
        self.attention = "general"
        self.device = 'cpu'
        self.eval_task = '2014'

if __name__ == "__main__":
    cfg = Config()
    print(len(cfg.train_files))
