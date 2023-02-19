import torch
import pickle
import spacy
from spacy.lang.en import English
from spacy.lang.de import German
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from preprocess import *
from mosestokenizer import *
from config import *
from sacrebleu.metrics import BLEU, CHRF, TER

cfg = Config()

''' loading pickles'''

en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id = lookup_table_maker(cfg)
print(f'en_vocab = {en_vocab[:15]}')
print(f'de_vocab = {de_vocab[:15]}')
print(f'id_to_en = {id_to_en[:15]}')
print(f'id_to_de = {id_to_de[:15]}')
print(f'de_pad = {de_to_id["<pad>"]} // de_unk = {de_to_id["<unk>"]} // de_SOS = {de_to_id["<s>"]} // de_EOS = {de_to_id["</s>"]}')
print(f'en_pad = {en_to_id["<pad>"]} // en_unk = {en_to_id["<unk>"]} // en_SOS = {en_to_id["<s>"]} // en_EOS = {en_to_id["</s>"]}')
print(f'en_vocab_size = {len(en_vocab)} // de_vocab_size = {len(de_vocab)}')

# print(de_to_id['multiple'])
# # test1 = torch.tensor([[1,3,2,4,5,0,0],
# #                       [5,7,6,5,9,11,3],
# #                       [2,6,3,0,0,0,0]])
# # print(test1)

# # test2 =  torch.nn.utils.rnn.pack_padded_sequence(test1,[5,7,3], batch_first = True, enforce_sorted = False)
# # print(test2)
# # test3, len_unpacked = torch.nn.utils.rnn.pad_packed_sequence(test2, batch_first = True)
# # print(test3)


# test1 = 'I am a good boy'
# test2 = ['I am a good boy']

# bleu = BLEU()
# print(bleu.sentence_score(test1,test2))
# print(bleu.corpus_score(test1,test2))

count = 0
with MosesTokenizer('en') as en_tokenizer:
    for vocab in en_vocab:
        preprocessed_vocab = vocab
        tokenized_vocab = en_tokenizer(vocab)
        if preprocessed_vocab != tokenized_vocab[0]:
            print(f'original_vocab = {preprocessed_vocab}')
            print(f'tokenized_vocab = {tokenized_vocab[0]}')
            count +=1

print(count)


