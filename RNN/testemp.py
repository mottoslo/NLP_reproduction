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

model_output = torch.randint(low = 0, high = 50001, size = (128,49))
cont_model = []
cont_target = []
for sent in model_output:
    words = [id_to_de[idx] for idx in sent]
    with MosesDetokenizer('en') as detokenize:
        detokenized = detokenize(words)
        cont_model.append(detokenized)

data = pickle.load(open('./data/train_indices_tuple_small', 'rb'))
_ , de_in_data, _ = data
for sent in de_in_data[:128]:
    words = [id_to_de[idx] for idx in sent]
    with MosesDetokenizer('en') as detokenize:
        detokenized = detokenize(words)
        cont_target.append(detokenized)

print(cont_model[:3])
print(cont_target[:3])

cont_model = ['I am an apple']
cont_target = ['I am an apple']
bleu = BLEU()
score = bleu.corpus_score(cont_model, cont_target)

print(f'bleu score corpus = {score}')





    
    

# score = bleu.corpus_score(de_in_train, de_out_train)
# print(score)




# lst = [torch.tensor(en_train[i]) for i in range(batch_size)]
# padded = nn.utils.rnn.pad_sequence(lst,batch_first = True)
# print(padded)


# de_in_train = train_data[1]
# de_out_train = train_data[2]


# print(f'de_train_in_data = {de_out_train[:5]}')
# print(f'de_in_train padded = {padded}')


# UNK_TOKEN = 1
# with open('data/en_train', 'r', encoding='utf-8') as f:
#     en_train = f.readlines()
#     sample = en_train[0]
#     print(f'sample = {sample}')
#     with MosesTokenizer('en') as tokenize_en:
#         tokens = tokenize_en(sample)
#         print(f'tokens = {tokens}')
#         as_indices = [en_to_id[token] if token in en_to_id else en_to_id['<unk>'] for token in tokens]
#         print(f'indices = {as_indices}')

# train_data = pickle.load(open('./data/train_indices_tuple_small', 'rb'))
# zipped = list(zip(train_data[0],train_data[1],train_data[2]))
# print(len(zipped))
# en_train, de_in_train, de_out_train = zip(*zipped)




