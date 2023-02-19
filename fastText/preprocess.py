import numpy as np
import time
import collections
from tqdm.auto import tqdm
import os
from utils import *
import pickle
import config

csv.field_size_limit(100000000)


def save_txt_label(raw_data_folder, cfg):

    tasks = os.listdir(raw_data_folder)
    for task in tasks:
        rows = readcsv(raw_data_folder + f'/{task}/train.csv')
        txt, label = text_and_label(rows)
        pickle.dump(txt, open(cfg.train_data_save_path + f'/{task}_text_train.pkl', 'wb'))
        pickle.dump(label, open(cfg.train_data_save_path + f'/{task}_label_train.pkl', 'wb'))
        print(f'pickle dumped for {task}_train')

        rows = readcsv(raw_data_folder + f'/{task}/test.csv')
        txt, label = text_and_label(rows)
        pickle.dump(txt, open(cfg.test_data_save_path + f'/{task}_text_test.pkl', 'wb'))
        pickle.dump(label, open(cfg.test_data_save_path + f'/{task}_label_test.pkl', 'wb'))
        print(f'pickle dumped for {task}_test')

def make_pickles(cfg):
    """create dictionary of {word : frequency} from all training files"""
    tasks = os.listdir(cfg.raw_data_folder)
    # files = cfg.train_files
    for task in tqdm(tasks, desc="Creating Vocabulary"):
        print(f'looking at {task}_train.pkl')
        vocabulary = {}
        with open(cfg.train_data_save_path + f'/{task}_text_train.pkl',"rb") as f:
            pkl = pickle.load(f)
            for lines in pkl:
                tokens = tokenize(lines)
                for token in tokens:
                    if token in vocabulary.keys():
                        vocabulary[token] +=1
                    else:
                        vocabulary[token] = 1

        vocabulary = sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)
        vocabulary = {vocab: i for vocab, i in vocabulary if i >= cfg.MIN_COUNT}
        
        pickle.dump(vocabulary, open(cfg.pickle_save_path + f'/vocabulary_{task}.pkl', 'wb'))

        word_to_index = create_word_to_index(vocabulary)
        pickle.dump(word_to_index, open(cfg.pickle_save_path + f'/word_to_index_{task}.pkl', 'wb'))

        index_to_word = create_index_to_word(word_to_index)
        pickle.dump(index_to_word, open(cfg.pickle_save_path + f'/index_to_word_{task}.pkl', 'wb'))


# def make_bigram_pickles(cfg):
#     """create dictionary of {word : frequency} from all training files"""
#     tasks = os.listdir(cfg.raw_data_folder)
#     # files = cfg.train_files
#     for task in tqdm(tasks, desc="Creating Bigram_Vocabulary"):
#         print(f'looking at {task}_train.pkl')
#         bigram_vocabulary = {}
#         word_to_index = pickle.load(open(cfg.pickle_save_path + f'/word_to_index_{task}.pkl', 'rb'))
#         with open(cfg.train_data_save_path + f'/{task}_text_train.pkl',"rb") as f:
#             pkl = pickle.load(f)
#             for lines in pkl:
#                 tokens = tokenize(lines)
#                 if len(tokens) < 2:
#                     print(f'{task} data with length {len(tokens)}')
#                 for i in range(len(tokens) - 1):
#                     temp_key = tokens[i] + ' ' + tokens[i+1]
#                     if (temp_key) in bigram_vocabulary.keys():
#                         bigram_vocabulary[temp_key] +=1
#                     else:
#                         bigram_vocabulary[temp_key] = 1

#         bigram_vocabulary = sorted(bigram_vocabulary.items(), key=lambda item: item[1], reverse=True)
#         bigram_vocabulary = {vocab: i for vocab, i in bigram_vocabulary}

#         pickle.dump(bigram_vocabulary, open(cfg.pickle_save_path + f'/bigram_vocabulary_{task}.pkl', 'wb'))

#         bigram_word_to_index = create_word_to_index(bigram_vocabulary)
#         pickle.dump(bigram_word_to_index, open(cfg.pickle_save_path + f'/bigram_word_to_index_{task}.pkl', 'wb'))

#         bigram_index_to_word = create_index_to_word(bigram_word_to_index)
#         pickle.dump(bigram_index_to_word, open(cfg.pickle_save_path + f'/bigram_index_to_word_{task}.pkl', 'wb'))

def make_bigram_pickles(cfg):
    """create dictionary of {word : frequency} from all training files"""
    tasks = os.listdir(cfg.raw_data_folder)
    # files = cfg.train_files
    bigram_vocabulary = {}
    for task in tqdm(tasks, desc="Creating Bigram_Vocabulary"):
        print(f'looking at {task}_train.pkl')
        # word_to_index = pickle.load(open(cfg.pickle_save_path + f'/word_to_index_{task}.pkl', 'rb'))
        with open(cfg.train_data_save_path + f'/{task}_text_train.pkl',"rb") as f:
            pkl = pickle.load(f)
            for lines in pkl:
                tokens = tokenize(lines)
                # if len(tokens) < 2:
                    # print(f'{task} data with length {len(tokens)}')
                for i in range(len(tokens) - 1):
                    temp_key = tokens[i] + ' ' + tokens[i+1]
                    if (temp_key) in bigram_vocabulary.keys():
                        bigram_vocabulary[temp_key] +=1
                    else:
                        bigram_vocabulary[temp_key] = 1

    bigram_vocabulary = sorted(bigram_vocabulary.items(), key=lambda item: item[1], reverse=True)
    bigram_vocabulary = {vocab: i for vocab, i in bigram_vocabulary}

    pickle.dump(bigram_vocabulary, open(cfg.pickle_save_path + f'/bigram_vocabulary.pkl', 'wb'))

    bigram_word_to_index = create_word_to_index(bigram_vocabulary)
    pickle.dump(bigram_word_to_index, open(cfg.pickle_save_path + f'/bigram_word_to_index.pkl', 'wb'))

    bigram_index_to_word = create_index_to_word(bigram_word_to_index)
    pickle.dump(bigram_index_to_word, open(cfg.pickle_save_path + f'/bigram_index_to_word.pkl', 'wb'))


def create_word_to_index(vocabulary):
    """create word_to_idx dictionary from vocab"""
    word_to_index = {word: i for i, word in enumerate(vocabulary.keys())}
    return word_to_index

def create_index_to_word(word_to_index):
    """create word_to_idx dictionary from vocab"""
    index_to_word = list(word_to_index.keys())
    return index_to_word

    


if __name__ == '__main__':
    cfg = config.Config()
    # save_txt_label(cfg.raw_data_folder,cfg)
    # make_pickles(cfg)
    make_bigram_pickles(cfg)
    # pickle.dump(vocabulary, open(cfg.pickle_save_path + '/vocabulary.pkl', 'wb'))

    # word_to_index = create_word_to_index(vocabulary)
    # pickle.dump(word_to_index, open(cfg.pickle_save_path + '/word_to_index.pkl', 'wb'))

    # index_to_word = create_index_to_word(word_to_index)
    # pickle.dump(index_to_word, open(cfg.pickle_save_path + '/index_to_word.pkl', 'wb'))

    # tasks = os.listdir(cfg.raw_data_folder)
    # for task in tasks:

    #     vocabulary = pickle.load(open(cfg.pickle_save_path + f'/vocabulary_{task}.pkl', 'rb'))
    #     word_to_index = pickle.load(open(cfg.pickle_save_path + f'/word_to_index_{task}.pkl', 'rb'))
    #     index_to_word = pickle.load(open(cfg.pickle_save_path + f'/index_to_word_{task}.pkl', 'rb'))
    #     print(f'{task} : {index_to_word[-10:]}')
    
    # make_pickles(cfg)
    # make_bigram_pickles(cfg)

    


