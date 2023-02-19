import numpy as np
import time
import collections
from tqdm.auto import tqdm
import re
import csv
csv.field_size_limit(100000000)

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
        temp = temp + row[1]
        i = 2
        while(i <len(row)):
            temp = temp + row[i]
            i += 1

        txt.append(clean_str(temp))
            
    return txt, label

def tokens_to_indices(tokens,word_to_idx,vocabulary):
    indices = []
    for token in tokens:
        if token in vocabulary.keys():
            indices.append(word_to_idx[token])
        else:
            indices.append(len(vocabulary))   #oov token

    return indices

def bigram_tokens_to_indices(tokens, bigram_word_to_idx, bigram_vocabulary):
    indices = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + ' ' + tokens[i + 1]
        if bigram in bigram_vocabulary.keys():
            indices.append(bigram_word_to_idx[bigram])
        # else:
            # indices.append(len(bigram_vocabulary))   #oov token
            
    return indices



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

    



# def create_vocab():
#     """create dictionary of {word : frequency} from all training files"""
#     vocabulary = {}
#     files = cfg.train_files
#     for i, file in tqdm(enumerate(files), desc="Creating Vocabulary", total=len(files), ncols=70):
#         print(f'file{i+1}')
#         with open(file,"rb") as f:
#             pkl = pickle.load(f)
#             for lines in pkl:
#                 tokens = tokenize(lines)
#                 for token in tokens:
#                     if token in vocabulary.keys():
#                         vocabulary[token] +=1
#                     else:
#                         vocabulary[token] = 1

#     vocabulary = sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)
#     vocabulary = {vocab: i for vocab, i in vocabulary if i >= cfg.MIN_COUNT}
#     # vocabulary = {vocab: i for vocab, i in vocabulary if i >= cfg.MIN_COUNT}  # vocab freq < MIN_COUNT removed

#     return vocabulary



