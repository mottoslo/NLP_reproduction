import numpy as np
import re
import collections
import torch
import pickle
from config import *
from tqdm.auto import tqdm

def lookup_table_maker(cfg):
    with open(cfg.en_vocab_path, 'r', encoding='utf-8') as f:
        en_vocab = [word.strip('\n') for word in f.readlines()]
        id_to_en = ['<pad>'] + en_vocab
        en_to_id = {x : i for i, x in enumerate(id_to_en)}
    with open(cfg.de_vocab_path, 'r', encoding='utf-8') as f:
        de_vocab = [word.strip('\n') for word in f.readlines()]
        id_to_de = ['<pad>'] + de_vocab
        de_to_id = {x : i for i, x in enumerate(id_to_de)}
    return en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id

def load_data(en_file, de_file, en_to_id, de_to_id):
    with open(en_file, 'r', encoding='utf-8') as f:
        en_file = f.readlines()
    with open(de_file, 'r', encoding='utf-8') as k:
        de_file = k.readlines()
    en_container = []
    de_in_container = []
    de_out_container = []
    for i in tqdm(range(len(en_file)), desc = 'loading data...', ncols = 70):
    # for i in tqdm(range(10), desc = 'loading data...', ncols = 70):
        en_tokenized = en_file[i].strip('\n').split()
        de_tokenized = de_file[i].strip('\n').split()
        if (len(en_tokenized)>50) or (len(de_tokenized)>50): continue
        if (len(en_tokenized) == 0) or (len(de_tokenized) == 0): continue
        en_indices = [en_to_id[token] if token in en_to_id else en_to_id['<unk>'] for token in en_tokenized]
        de_indices = [de_to_id[token] if token in de_to_id else de_to_id['<unk>'] for token in de_tokenized]
        en_container.append(en_indices)
        de_in_container.append([de_to_id['<s>']] + de_indices)
        de_out_container.append(de_indices + [de_to_id['</s>']])

    print(f'en_indices loaded with sent num = {len(en_container)}')
    print(f'de_indices loaded with sent num = {len(de_in_container)}')

    return en_container, de_in_container, de_out_container


if __name__ == '__main__':
    cfg = Config()
    en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id = lookup_table_maker(cfg)

    # train_data = load_data(cfg.en_train_path, cfg.de_train_path, en_to_id, de_to_id)
    # test_data = load_data(cfg.en_test_path, cfg.de_test_path, en_to_id, de_to_id)
    # val_data = load_data(cfg.en_val_path, cfg.de_val_path, en_to_id, de_to_id)
    # pickle.dump(train_data, open(cfg.data_folder_path + 'train_indices_tuple', 'wb'))
    # pickle.dump(test_data, open(cfg.data_folder_path + 'test_indices_tuple', 'wb'))
    # pickle.dump(val_data, open(cfg.data_folder_path + 'val_indices_tuple', 'wb'))
    print(id_to_en[:5])
    print(id_to_de[:5])
    print(len(en_vocab))
    




    