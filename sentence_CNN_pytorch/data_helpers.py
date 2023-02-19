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

def load_trec_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [" ".join(sent.split()[2:]) for sent in train]
    container_y = []

    for sent in train:
        if sent[0] == 'a':
            container_y.append(0)
        elif sent[0] == 'd':
            container_y.append(1)
        elif sent[0] == 'e':
            container_y.append(2)
        elif sent[0] == 'h':
            container_y.append(3)
        elif sent[0] == 'l':
            container_y.append(4)
        elif sent[0] == 'n':
            container_y.append(5)
        #container_x.append(" ".join(sent.split()[2:]))
    return container_x, container_y

def load_mr_data(pos_file, neg_file):
    pos_text = list(open(pos_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    pos_text = [clean_str(sent) for sent in pos_text] # clean_str 함수로 전처리 (소문자, 특수 기호 제거, (), 등 분리)


    neg_text = list(open(neg_file, "r", encoding='latin-1').readlines()) # 부정적인 review 읽어서 list 형태로 관리
    neg_text = [clean_str(sent) for sent in neg_text]

    positive_labels = [1 for _ in pos_text] # 긍정 review 개수만큼 ground_truth 생성 [0, 1]
    negative_labels = [0 for _ in neg_text] # 부정 review 개수만큼 ground_truth 생성 [0, 1]
    y = positive_labels + negative_labels

    x_final = pos_text + neg_text
    return [x_final, y]

def load_cr_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [sent[2:] for sent in train]
    container_y = []

    for sent in train:
        container_y.append(int(sent[0]))
    return container_x, container_y

def load_sst1_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [sent[2:] for sent in train]
    container_y = []

    for sent in train:
        container_y.append(int(sent[0]))
    return container_x, container_y

def load_sst2_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [sent[2:] for sent in train]
    container_y = []

    for sent in train:
        container_y.append(int(sent[0]))
    return container_x, container_y

def load_subj_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [sent[2:] for sent in train]
    container_y = []

    for sent in train:
        container_y.append(int(sent[0]))
    return container_x, container_y

def load_mpqa_data(train_file):
    train = list(open(train_file, "r", encoding='latin-1').readlines()) # 긍정적인 review 읽어서 list 형태로 관리
    train = [clean_str(sent) for sent in train]
    container_x = [sent[2:] for sent in train]
    container_y = []

    for sent in train:
        container_y.append(int(sent[0]))
    return container_x, container_y

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

def text_to_indices(x_text, word_id_dict, use_unk=False):
    text_indices = []

    for text in x_text:
        words = text.split()
        ids = [2]  # <s>
        for word in words: # i, am, a, boy
            if word in word_id_dict:
                word_id = word_id_dict[word]
            else:  # oov
                if use_unk:
                    word_id = 1 # OOV (out-of-vocabulary)
                else:
                    word_id = len(word_id_dict)
                    word_id_dict[word] = word_id
            ids.append(word_id) # 5, 8, 6, 19
        ids.append(3)  # </s>
        text_indices.append(ids)
    return text_indices

def sequence_to_tensor(sequence_list, nb_paddings=(0, 0)):
    nb_front_pad, nb_back_pad = nb_paddings

    max_length = len(max(sequence_list, key=len)) + nb_front_pad + nb_back_pad
    sequence_tensor = torch.LongTensor(len(sequence_list), max_length).zero_()  # 0: <pad>
    print("\n max length: " + str(max_length))
    for i, sequence in enumerate(sequence_list):
        sequence_tensor[i, nb_front_pad:len(sequence) + nb_front_pad] = torch.tensor(sequence)
    return sequence_tensor