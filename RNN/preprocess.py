import numpy as np
import re
import collections
import torch

def clean_str(string):

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def buildVocab(sentences, vocab_size): 
    # Build vocabulary
    words = []
    for sentence in sentences: words.extend(sentence.split()) # i, am, a, boy, you, are, a, girl
    print("The number of words: ", len(words))
    word_counts = collections.Counter(words)
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    # vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)} # a: 0, i: 1...
    return [vocabulary, vocabulary_inv]

import numpy as np
from tqdm.auto import tqdm
import pickle
from mosestokenizer import *
from config import *

#Read Vocabulary
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

#Make Dataset
def load_data(en_file, de_file, en_to_id, de_to_id):
    with open(en_file, 'r', encoding='utf-8') as f:
        en_file = f.readlines()
    with open(de_file, 'r', encoding='utf-8') as k:
        de_file = k.readlines()
    en_container = []
    de_in_container = []
    de_out_container = []

    with MosesTokenizer('en') as en_tokenizer:
        with MosesTokenizer('de') as de_tokenizer:
            for i in tqdm(range(len(en_file)), desc = 'loading data...', ncols = 70):
            # for i in tqdm(range(100), desc = 'loading data...', ncols = 70):
                en_tokenized = en_tokenizer(en_file[i])
                de_tokenized = de_tokenizer(de_file[i])
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

    # id_to_en, en_to_id, id_to_de, de_to_id = read_vocab()
    # x_train, y_train_in, y_train_out = load_data(['data/en_train', 'data/de_train'], en_to_id, de_to_id)
    # x_train, y_train_in, y_train_out = make_tensor(x_train), make_tensor(y_train_in), make_tensor(y_train_out)

    # x_test, y_test_in, y_test_out = load_data(['data/en_test2014', 'data/de_test2014'], en_to_id, de_to_id)
    # x_test = make_tensor(x_test)

    # data = {'vocab':(id_to_en, en_to_id, id_to_de, de_to_id), 'train':(x_train, y_train_in, y_train_out), 'test':(x_test, y_test_in, y_test_out)}

    # with open('data/train_test_tensor', 'wb') as f:
    #     pickle.dump(data, f)
    '''loading data'''
    en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id = lookup_table_maker(cfg)
    train_data = load_data(cfg.en_train_path, cfg.de_train_path, en_to_id, de_to_id)
    test_data = load_data(cfg.en_test_path, cfg.de_test_path, en_to_id, de_to_id)
    val_data = load_data(cfg.en_val_path, cfg.de_val_path, en_to_id, de_to_id)
    pickle.dump(train_data, open(f'./data/train_indices_tuple', 'wb'))
    pickle.dump(test_data, open(f'./data/test_indices_tuple', 'wb'))
    pickle.dump(val_data, open(f'./data/val_indices_tuple_2015', 'wb'))


    

    # train_data = pickle.load(open(f'./data/train_indices_tuple', 'rb'))
    # val_data = pickle.load(open(f'./data/val_indices_tuple', 'rb'))
    # print(train_data[0][:7])
    # print(train_data[1][:7])
    # print(val_data[0][:7])
    # print(val_data[1][:7])


