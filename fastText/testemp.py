import csv
import numpy as np
from config import Config
from utils import *
from tqdm.auto import tqdm
import collections
import pickle
import math
import random
import time
import pandas as pd

config = Config()


def readcsv(file_path):
    file = open(file_path)
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
            rows.append(row)
    return rows


def text_and_label(rows):
    label = []
    txt = []
    for row in rows:
        temp = ''
        label.append(int(row[0]))
        temp = temp + ' ' + row[1]
        i = 2
        while(i <len(row)):
            temp = temp + ' ' + row[i]
            i += 1

        txt.append(clean_str(temp))
            
    return txt, label


if __name__ == '__main__':
    cfg = Config()
    # tasks = cfg.num_classes.keys()
    # for task in tasks:
    #     dat = pd.read_csv(cfg.raw_data_folder + f'/{task}/train.csv')
    #     dat['new'] = dat[dat.columns[1:]].agg(' '.join, axis=1)
    #     label = dat[dat.columns[0]].to_numpy(dtype = int)
    #     text = dat[dat.columns[1]].to_numpy(dtype = str)

    #     pickle.dump(label, open(cfg.train_data_save_path + f'/{task}_label_train.pkl', 'wb'))
    #     pickle.dump(text, open(cfg.train_data_save_path + f'/{task}_text_train.pkl', 'wb'))
    #     print(f'pickle dumped for {task}')
        
    # sample = [[3, 'he', 'was', 'a',' boy',' she', 'was', 'a', 'girl'],[3, 'he', 'was', 'a',' boy',' she', 'was', 'a', 'girl']]
    # txt, label = text_and_label(sample)
    # print(txt)
    # print(label)


    # dat = dat[[dat.columns[0], 'new']]
    # print(f'shape : {dat.shape}')
    # print(dat)

    
    # vocab = collections.Counter()
    # vocab['a'] = 1
    # vocab['b'] = 2
    # vocab['c'] = 3
    # vocab = sorted(vocab.items(), reverse=True)
    # vocab = {voca: i for voca, i in vocab if i >= 0}
    # print(vocab)
    # vocab2 = {'a' : 1, 'b' : 5, 't' : 123}
    
    # for task in cfg.num_classes.keys():

    #     testing = pickle.load(open(cfg.train_data_save_path + f'/{task}_label_train.pkl', 'rb'))
    #     train_testing = pickle.load(open(cfg.train_data_save_path + f'/{task}_text_train.pkl', 'rb'))
    #     min_length = math.inf
    #     max_length = 0
    #     zero_counter = 0
    #     sent_counter = 0
    #     sent_len_counter = 0
    #     for sent in train_testing:
    #         tokens = tokenize(sent)
    #         sent_len = len(tokens)
    #         sent_len_counter += sent_len
    #         if sent_len > max_length:
    #             max_length = sent_len
    #         if sent_len < min_length:
    #             min_length = sent_len
    #         if sent_len < 2:
    #             zero_counter += 1
    #         else:
    #             sent_counter +=1
    #     print(f'{task} : len {len(train_testing)} min {min_length} max {max_length} num zero {zero_counter} agv sent len : {sent_len_counter / sent_counter}')
        # cont = {}
        # for sample in testing:
        #     if sample not in cont.keys():
        #         cont[sample] = 1
        #     if sample in cont.keys():
        #         cont[sample] +=1
        # print(f'labels in {task} : {cont}')


    # testing = pickle.load(open(cfg.train_data_save_path + f'/ag_label_train.pkl', 'rb'))
    # train_testing = pickle.load(open(cfg.train_data_save_path + f'/ag_text_train.pkl', 'rb'))
    # start_time = time.time()
    # zipped = list(zip(testing,train_testing))
    # random.shuffle(zipped)


    # testing = pickle.load(open(cfg.train_data_save_path + f'/yelpf_text_train.pkl', 'rb'))
    # small = 0
    # large = 0
    # empty = 0
    # super_large = 0
    # for sent in testing:
    #     tokens = tokenize(sent)
    #     length = len(tokens)
    #     if length < 10:
    #         small +=1
    #         if length <2:
    #             empty +=1
    #     elif length > 500:
    #         large +=1
    #         if length >1000:
    #             super_large +=1
    
    # print(f'small = {small}\nlarge = {large}\nempty = {empty}\nsuper_large = {super_large}')


    # label = pickle.load(open(cfg.train_data_save_path + f'/yelpp_label_train.pkl', 'rb'))
    # text = pickle.load(open(cfg.train_data_save_path + f'/yelpp_text_train.pkl', 'rb'))

    # for i, txt in enumerate(text):
    #     tokens = tokenize(txt)
    #     if len(tokens) < 2:
    #         print(f'label : {label[i]}')
    #         print(f'txt : {tokens}\n')
    bi_vocab = pickle.load(open(cfg.pickle_save_path + f'/bigram_vocabulary_amzf.pkl', 'rb'))
    print(bi_vocab.keys())

