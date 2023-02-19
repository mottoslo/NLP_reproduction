
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
import math
from tqdm.auto import tqdm
import numpy as np
# import heapq
import pickle
import random
from config import Config
import os
import time
import re
import heapq
# import nltk

cfg = Config()

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

def tokenize(sent):
    """token만들기
       list로 반환"""

    tokens = []
    token = ''
    for c in sent:
        if c == '\r':
            continue
        if c == ' ' or c == '\t' or c == '\n':
            if len(token) >= 100:
                token = token[:100]
            tokens.append(token)
            token = ''
            # if c == '\n':
            #     tokens.append('</s>')
        else:
            token += c
    if token != '':
        tokens.append(token)
    return tokens

def create_vocab():
    """create dictionary of {word : frequency} from all training files"""
    vocabulary = {}
    files = cfg.train_files
    for i, file in tqdm(enumerate(files), desc="Creating Vocabulary", total=len(files), ncols=70):
        print(f'file{i+1}')
        with open(file,"rb") as f:
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
    print(len(table))
    return table

# def create_huffman_tree():

def make_pickles():

    vocabulary = create_vocab()
    pickle.dump(vocabulary, open(cfg.vocabulary_path, 'wb'))

    word_to_index = create_word_to_index(vocabulary)
    pickle.dump(word_to_index, open(cfg.word_to_index_path, 'wb'))

    index_to_word = create_index_to_word(word_to_index)
    pickle.dump(index_to_word, open(cfg.index_to_word_path, 'wb'))

    unigram_table = create_unigram_table(vocabulary, word_to_index, 0.75)
    pickle.dump(unigram_table, open(cfg.unigram_table_path, 'wb'))

    #make huffman tree
    tree = huffman_tree(vocabulary)
    tree = heapq.heappop(tree)
    pickle.dump(tree, open(cfg.huffman_tree_path, 'wb'))
    #make path 
    bin_paths = np.empty(shape = len(vocabulary))
    mover(tree, 0b1)
    pickle.dump(bin_paths, open(cfg.bin_paths_path, 'wb'))
    
    #make index_container
    # for i in range(len(bin_paths)):
        # print(bin(int(bin_paths[i])))
    index_container = idx_maker(tree, bin_paths, {})
    pickle.dump(index_container, open(cfg.index_container_path, 'wb'))
    
    
    # huffman_tree = create_huffman_tree(files_path)
    # pickle.dump(huffman_tree, open(cfg.huffman_tree_path), 'wb')            =>어떤 argument 와야하는지 확인

def subsampling(word_idx, vocabulary, index_to_word, total, threshold):
    """논문 subsampling 식에 의거하여 word를 포함하면 True, 제외하면 False를 리턴"""
    freq_ratio = vocabulary[index_to_word[word_idx]] / total
    discard_prob = 1 - np.sqrt((threshold / freq_ratio))
    rand = np.random.uniform()
    decision = True if rand > discard_prob else False
    word = index_to_word[word_idx]
    # print(f'word : {word} // freq_ratio : {freq_ratio} // discard_prob : {discard_prob} // draw : {rand} // decision : {decision}')

    return decision




def make_inout_pair(tokenized, window_size, word_to_index, vocabulary, index_to_word, total, threshold):
    '''word가 들어오면 index로 변환해서 진행'''

    token_to_index = []
    for word in tokenized:
        if word in word_to_index.keys():
            token_to_index.append(word_to_index[word])


    input = []
    output = []
    sent_len = len(token_to_index)

    if cfg.model_type == "SG" or "CBOW":
        """input이 list [] // output이 list in list [ [indices1], [indices2], [indices3], [indices4] ]"""
        # for i in range(1 + window_size, sent_len - window_size):
        #     input.append(tokenized[i])
        #     for j in reversed(range(1, window_size)):
        #         output.append(tokenized[i - j])
        #     for k in range(1,window_size):
        #         output.append(tokenized[i + j])
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

    # if cfg.model_type == "asdfasf":
    #     """output이 list [] // input이 list in list [ [indices1], [indices2], [indices3], [indices4] ]"""
    #     # for i in range(1 + window_size, sent_len - window_size):
    #     #     input.append(tokenized[i])
    #     #     for j in reversed(range(1, window_size)):
    #     #         output.append(tokenized[i - j])
    #     #     for k in range(1,window_size):
    #     #         output.append(tokenized[i + j])

    #     for center_word_idx in range(sent_len):
    #         # if token_to_index[center_word_idx] not in word_to_index.keys():
    #         #     continue       => token_to_index에서 다시 word로 바꿔서 찾아야함
    #         output.append(token_to_index[center_word_idx])
    #         rand_window_size = random.randint(1, window_size + 1)
    #         context_idx = list(range(max(0, center_word_idx - rand_window_size), center_word_idx)) \
    #                       + list(range(center_word_idx + 1, min(sent_len, center_word_idx + rand_window_size + 1)))
    #         input_temp = [token_to_index[k] for k in context_idx]
    #         input.append(input_temp)

    return input, output

def preprocess(train_file, window_size, word_to_index, vocabulary, index_to_word, total, threshold):
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
    # print([index_to_word[i] for i in input])
    return input, target



def huffman_tree(vocabulary):
    """
    frequency: list of elements (word, frequency), ordered by frequency from max to min
    """
    length = len(vocabulary)
    # use index to prevent error of comparing int and string
    heap = [[item[1], i] for i, item in enumerate(vocabulary.items())]     # [frequency, vocab_idx]
    heapq.heapify(heap)
    for i in tqdm(range(length - 1), desc="Creating Huffman Tree", ncols=70):
        min1 = heapq.heappop(heap)
        min2 = heapq.heappop(heap)
        heapq.heappush(heap, [min1[0] + min2[0], i + length, min1, min2])
    
    return heap

    # node of heap : [frequency, index, left child, right child]

def mover(heap, path_counter):  # path_counter는 0b0에서 시작
    if len(heap) > 2:   #leaf 아닐때
        mover(heap[2], (path_counter << 1))    #왼쪽으로 이동
        mover(heap[3], (path_counter << 1) + 0b1)    #오른쪽으로 이동
    elif len(heap) <= 2:   #leaf
        bin_paths[heap[1]] = path_counter

def idx_maker(heap, bin_paths, index_container):
    for i in tqdm(range(len(bin_paths)), desc = 'making node_indexes...'):
        current_heap = heap
        # print(bin_paths[i])
        path = str(bin(int(bin_paths[i])))
        # print(f'{i}th path : {path[3:]}')
        path_counter = 3
        temp_container = []
        # print(f'i : {i}')
        while(len(current_heap) > 2):
            temp_container.append(current_heap[1])
            if path[path_counter] == '1':
                current_heap = current_heap[3]
            elif path[path_counter] == '0':
                current_heap = current_heap[2]
            path_counter +=1
        index_container[current_heap[1]] = temp_container
    return index_container

def path_to_arr(num):
    bnr = bin(num).replace('0b1','')
    bnr = np.array([int(i) for i in bnr])
    # bnr[bnr == 0] = -1

    return bnr
    
        
            
        
        

        

        
        
    




if __name__ == "__main__":
    # create_vocab()


    # make_pickles()
    with open(cfg.vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)
    with open(cfg.word_to_index_path, 'rb') as f:
        word_to_index = pickle.load(f)
    with open(cfg.index_to_word_path, 'rb') as f:
        index_to_word = pickle.load(f)
    with open(cfg.unigram_table_path, 'rb') as f:
        unigram_table = pickle.load(f)
    # with open(cfg.huffman_tree_path, 'rb') as f:
        # huffman_tree = pickle.load(f)
    # with open(cfg.bin_paths_path, 'rb') as f:
        # bin_paths = pickle.load(f)
    # with open(cfg.index_container_path, 'rb') as f:
        # index_container = pickle.load(f)
        
    # print(len(index_container))
    # print(len(vocabulary))
    # maxi = 0
    # mini = 10000000
    # for i in range(330657):
    #     tempmax = max(index_container[i])
    #     tempmin = min(index_container[i])
    #     if int(tempmax) > maxi:
    #         maxi = tempmax
    #     if tempmin < mini:
    #         mini = tempmin
    # print(tempmax)
    # print(tempmin)


    # total = sum(vocabulary.values())
    # print(total)
    # print(vocabulary)
    # print(word_to_index)
    # print(index_to_word)
    # print(unigram_table)
    # temp_input, temp_output = preprocess(cfg.train_files[1], 5, word_to_index, vocabulary, index_to_word, total)
    # print("----------")
    # print(temp_input[0])
    # print(temp_output[0])

    # for filename in os.listdir(cfg.train_folder_path):
    #     with open(f'{cfg.train_folder_path}{filename}', 'r') as f:
    #         lines = f.readlines()
    #         lines = [clean_str(i) for i in lines]
    #         pickle.dump(lines, open(f'./data/clean/{filename}', 'wb'))
    #         print(f'pickle dumped for {filename}')

