from config import *
import numpy as np
import pickle
from utils import *
from tqdm.auto import tqdm
import math
import time
import random


def create_vocab():
    """create dictionary of {word : frequency} from all training files"""
    vocabulary = {}
    files = cfg.train_file
    # for i, file in tqdm(enumerate(files), desc="Creating Vocabulary", total=len(files), ncols=70):
    #     print(f'file{i+1}')
    with open(files, "rb") as f:
        pkl = pickle.load(f)
        for lines in tqdm(pkl, desc = 'creating vocabulary'):
            tokens = tokenize(lines)

            for token in tokens:
                if token in vocabulary.keys():
                    vocabulary[token] +=1
                else:
                    vocabulary[token] = 1



    vocabulary = sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)
    vocabulary = {vocab: i for vocab, i in vocabulary if i >= cfg.MIN_COUNT}
    # vocabulary = {vocab: i for vocab, i in vocabulary if i >= cfg.MIN_COUNT}  # vocab freq < MIN_COUNT removed

    return vocabulary

def create_word_to_index(vocabulary):
    """create word_to_idx dictionary from vocab"""
    word_to_index = {word: i for i, word in enumerate(vocabulary.keys())}
    return word_to_index

def create_index_to_word(word_to_index):
    """create word_to_idx dictionary from vocab"""
    index_to_word = list(word_to_index.keys())
    return index_to_word

def create_ngram_vocab(vocabulary):
    """create dictionary of {word : frequency} from all training files"""
    ngram_vocab = {}
    # for i, file in tqdm(enumerate(files), desc="Creating Vocabulary", total=len(files), ncols=70):
    #     print(f'file{i+1}')
    for word in tqdm(vocabulary.keys(), desc = 'creating ngram vocabulary'):
        #!!!!!!!!!!!!!!!!!!!!1
        ngrams = readWord_all_ngrams(word, n_gram_min = cfg.min_ngram, n_gram_max = cfg.max_ngram)
        
        for ngram in ngrams:
            if ngram not in ngram_vocab.keys():
                ngram_vocab[ngram] =1
    
    return ngram_vocab

def create_unigram_table(vocabulary, word_to_index, power = 0.75):
    """
    Return a uni-gram table from the index of word to its probability of appearance.
    P(w) = count(w)^power / sum(count^power)
    """
    print("creating unigram table.....")
    table = []
    for word in vocabulary.keys():

        adjusted_freq = int(math.pow(vocabulary[word], power))
        idx = word_to_index[word]
        table.extend([idx] * adjusted_freq)
    print(f'length of table = {len(table)}')
    return table

def make_inout_pair(tokenized, window_size, word_to_index, vocabulary, index_to_word, threshold, total):
    '''word가 들어오면 index로 변환해서 진행'''

    token_to_index = []
    for word in tokenized:
        if word in word_to_index.keys():
            token_to_index.append(word_to_index[word])

    input = []
    output = []
    sent_len = len(token_to_index)

    for center_word_idx in range(sent_len):
        # if token_to_index[center_word_idx] not in word_to_index.keys():
        #     continue       #=> token_to_index에서 다시 word로 바꿔서 찾아야함
        word_idx = token_to_index[center_word_idx]

        if subsampling(word_idx, vocabulary, index_to_word, total, threshold):          #index 넣어서 subsampling해줌
            # print(index_to_word[center_word_idx])
            input.append(word_idx)

            rand_window_size = random.randint(1, window_size + 1)
            context_idx = list(range(max(0, center_word_idx - rand_window_size), center_word_idx)) \
                        + list(range(center_word_idx + 1, min(sent_len, center_word_idx + rand_window_size + 1)))
            out_temp = [token_to_index[k] for k in context_idx]
            output.append(out_temp)

    return input, output

def preprocess(train_file, window_size, word_to_index, vocabulary, index_to_word, threshold, total):
    start_time = time.time()
    input = []
    target = []

    with open(train_file, 'rb') as f:
        f = pickle.load(f)
        for sent in tqdm(f, desc = f"generating train inout pair for {train_file}"):
            tokenized = tokenize(sent)
            in_words, out_words = make_inout_pair(tokenized,window_size,word_to_index, vocabulary, index_to_word, total,threshold)
            input.extend(in_words)
            target.extend(out_words)

    print("train inout pair generated!")
    print(f"time taken : {time.time() - start_time}")
    return input, target

def make_word_to_ngram_indices(cfg, vocabulary, word_to_index, ngram_word_to_index):
    cont = {}
    for word in vocabulary.keys():
        word_idx = word_to_index[word]
        ngrams = readWord_all_ngrams(word, n_gram_min = cfg.min_ngram, n_gram_max = cfg.max_ngram)
        ngram_ids = [ngram_word_to_index[ngram] for ngram in ngrams]
        cont[word_idx] = ngram_ids

    return cont


if __name__ == '__main__':
    cfg = Config()
    # vocabulary = create_vocab()
    # pickle.dump(vocabulary, open('pickles/vocabulary', 'wb'))
    # print('vocab_created')
    # word_to_index = create_word_to_index(vocabulary)
    # pickle.dump(word_to_index, open('pickles/word_to_index', 'wb'))
    # print('word_to_index created')
    # index_to_word = create_index_to_word(word_to_index)
    # pickle.dump(index_to_word, open('pickles/index_to_word', 'wb'))
    # print('index_to_word created')
    # unigram_table = create_unigram_table(vocabulary, word_to_index, power = 0.75)
    # pickle.dump(unigram_table, open('pickles/unigram_table', 'wb'))
    # print('unigram_table craeted!')

    # word_to_ngram_indices = make_word_to_ngram_indices(cfg, vocabulary, word_to_index, ngram_word_to_index)
    # pickle.dump(word_to_ngram_indices, open('./pickles/word_to_ngram_indices.pkl', 'wb'))


    vocabulary = pickle.load(open('pickles/vocabulary', 'rb'))
    word_to_index = pickle.load(open('pickles/word_to_index', 'rb'))
    index_to_word = pickle.load(open('pickles/index_to_word', 'rb'))
    total_words = sum(vocabulary.values())
    for i in range(50):
        center, context = preprocess(f'./data/split/wiki.en.{i}', cfg.max_window_size, word_to_index, vocabulary, \
        index_to_word, cfg.subsampling_threshold, total_words)
        pickle.dump(center, open(f'./data/split/wiki_center_words_{i}.pkl', 'wb'))
        pickle.dump(context, open(f'./data/split/wiki_context_words_{i}.pkl', 'wb'))


    # ngram_vocab = create_ngram_vocab(vocabulary)
    # pickle.dump(ngram_vocab, open('pickles/ngram_vocabulary', 'wb'))

    # ngram_word_to_index = create_word_to_index(ngram_vocab)
    # pickle.dump(ngram_word_to_index, open('pickles/ngram_word_to_index', 'wb'))
    # print('ngram_word_to_index created')
    # ngram_index_to_word = create_index_to_word(ngram_word_to_index)
    # pickle.dump(ngram_index_to_word, open('pickles/ngram_index_to_word', 'wb'))
    # print('ngram_index_to_word created')


    ''' full_txt 50개로 쪼개기'''
    # with open('./data/wiki.en.txt') as f:
    #     full_txt = f.readlines()
    #     full_len = len(full_txt)
    #     for i in range(50):
    #         chunk = full_txt[int(full_len * (i / 50)) : int(full_len * ((i+1) / 50))]
    #         pickle.dump(chunk, open(f'./data/split/wiki.en.{i}', 'wb'))
    
