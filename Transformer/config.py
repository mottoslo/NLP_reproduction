import os

class Config:
    def __init__(self):
        self.data_folder_path = '/home/user20/RNN/data/'
        self.en_vocab_path = self.data_folder_path + '/en_vocab'
        self.de_vocab_path = self.data_folder_path + '/de_vocab'
        self.en_train_path = self.data_folder_path + '/en_train'
        self.de_train_path = self.data_folder_path + '/de_train'
        self.en_val_path = self.data_folder_path + '/en_val'
        self.de_val_path = self.data_folder_path + '/de_val'
        self.en_test_path = self.data_folder_path + '/en_test2015'
        self.de_test_path = self.data_folder_path + '/de_test2015'
        self.model_save_path = f'./transformer_results'
        self.device = 'cpu'

        '''for model parameters'''
        self.warmup_steps = 4000
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.trg_sos_idx = 2
        self.enc_voc_size = 50001
        self.dec_voc_size = 50001
        self.d_model = 512
        self.n_head = 8
        self.max_len = 51
        self.ffn_hidden = 2048
        self.n_layers = 4
        self.drop_prob = 0.1
        self.adam_betas = [0.9, 0.98]
        self.device = None 
        self.batch_size = 64
        self.num_epochs = 10

        #self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                #  ffn_hidden, n_layers, drop_prob, device

if __name__ == "__main__":
    cfg = Config()
    print(len(cfg.train_files))
