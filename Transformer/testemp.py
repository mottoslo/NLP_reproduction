import os
import numpy as np
from tqdm.auto import tqdm
import pickle
import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from dataloader import *
from config import *
from tokenizers import SentencePieceBPETokenizer, Tokenizer

cfg = Config()
train_file = cfg.en_train_path

with open(train_file, 'r', encoding='utf-8') as f:
    en_file = f.readlines()
print(en_file[:10])

'''train & save tokenizer'''
# tokenizer = SentencePieceBPETokenizer()
# tokenizer.train_from_iterator(
#     en_file,
#     vocab_size=50000,
#     min_frequency=5,
#     show_progress=True,
#     limit_alphabet=500,
# )

# tokenizer.save('./vocab/bpetokenizer_test.json')

'''load tokenizer'''
loaded_tokenizer = Tokenizer.from_file('./vocab/bpetokenizer_test.json')

vocab = loaded_tokenizer.get_vocab()
print(sorted(vocab, key=lambda x: vocab[x])[:10])