import numpy
import csv
import pandas as pd
import pickle
from utils import *
from config import *

csv.field_size_limit(10000000000000)

# with open('data/wiki.en.txt') as f:
    # lines = f.readlines()[0].split(' ')
    # print(lines[0:10])

# with open('data/wiki.en.small.pkl', 'rb') as f:
#     txt_file = pickle.load(f)

# sent = txt_file[0]
# print(sent)
# tokens = tokenize(sent)
# print(tokens)

# print(readWord_all_ngrams('catastrophic',3,6))

# vocabulary = pickle.load(open('pickles/temp_vocabulary', 'rb'))
# print(vocabulary)
# print(len(vocabulary))

# with open('./data/wiki.en.txt') as f:
    # temp = f.readlines()
    # pickle.dump(temp, open('./data/wiki.en.pkl', 'wb'))

cfg = Config()
# word_to_index = pickle.load(open('pickles/word_to_index', 'rb'))
# index_to_word = pickle.load(open('pickles/index_to_word', 'rb'))
# ngram_word_to_index = pickle.load(open('pickles/ngram_word_to_index', 'rb'))
# ngram_word_to_index = pickle.load(open('pickles/ngram_word_to_index', 'rb'))
# ngram_index_to_word = pickle.load(open('pickles/ngram_index_to_word', 'rb'))
# print(len(ngram_vocabulary.items()))

# sum = 0
# for i in range(50):
#     txt = pickle.load(open(f'./data/split/wiki.en.{i}', 'rb'))
#     print(len(txt))
#     sum += len(txt)

# print(f'sum = {sum}')
# for i in range(7):
#     center = pickle.load(open(f'./data/split/wiki_center_words_{i}.pkl', 'rb'))
#     context = pickle.load(open(f'./data/split/wiki_center_words_{i}.pkl', 'rb'))
#     print(f'{len(center)} // {len(context)}')

# word_to_ngram_indices = pickle.load(open(cfg.pickle_path + '/word_to_ngram_indices', 'rb'))
# print(word_to_ngram_indices[10000])
# index_to_word = pickle.load(open('pickles/index_to_word', 'rb'))
# print(index_to_word[2])
# center = pickle.load(open(f'./data/split/wiki_center_words_0.pkl', 'rb'))
# context = pickle.load(open(f'./data/split/wiki_context_words_0.pkl', 'rb'))
# print(center[:10])
# print(context[:10])
# test1 = pickle.load(open(f'./data/split/wiki.en.0', 'rb'))
# print(test1[:10])

# ngram_word_to_index = pickle.load(open('pickles/ngram_word_to_index', 'rb'))
# word_to_index = pickle.load(open('pickles/word_to_index', 'rb'))
# print(ngram_word_to_index[''])

# word_to_ngram_indices = pickle.load(open(cfg.pickle_path + '/word_to_ngram_indices', 'rb'))
# sum = 0 
# for key in word_to_ngram_indices.keys():
#     len1 = len(word_to_ngram_indices[key])
#     if len1 == 0:
#         sum +=1
#         print(key)

vocabulary = pickle.load(open('pickles/vocabulary', 'rb'))
unigram_table = pickle.load(open('pickles/unigram_table', 'rb'))
ngram_word_to_index = pickle.load(open('pickles/ngram_word_to_index', 'rb'))

len_unigram_table = len(unigram_table)
total_words = sum(vocabulary.values())
vocab_size = len(vocabulary)
ngram_len = len(ngram_word_to_index)

print(f'len unigram = {len_unigram_table} // total_words = {total_words} // vocab_size = {len(vocabulary)} // ngram_len = {ngram_len}')

    

    
    

