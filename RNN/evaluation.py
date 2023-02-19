import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm.auto import tqdm
import random
from collections import Counter
import time
from layers import *
from config import *
import sys
from preprocess import *
from mosestokenizer import *
import sacrebleu
from sacrebleu.metrics import BLEU

def see_translation(cfg, model, id_to_en, id_to_de):

    val_data = pickle.load(open(cfg.data_folder_path + 'val_indices_tuple', 'rb'))
    en_val, de_in_val, de_out_val = val_data

    en_lst = [torch.tensor(en_line) for en_line in en_val[500 : 500 + cfg.batch_size]]
    en_sample = nn.utils.rnn.pad_sequence(en_lst,batch_first = True).to(cfg.device)

    de_in_lst = [torch.tensor(de_in_line) for de_in_line in de_in_val[500 : 500 + cfg.batch_size]]
    de_in_sample = nn.utils.rnn.pad_sequence(de_in_lst,batch_first = True).to(cfg.device)
    
    de_out_lst = [torch.tensor(de_out_line) for de_out_line in de_out_val[500 : 500 + cfg.batch_size]]
    de_out_sample = nn.utils.rnn.pad_sequence(de_out_lst,batch_first = True).to(cfg.device)
    bleu_score = BLEU(effective_order = True)


    model.eval()
    forwarded = model(en_sample, de_in_sample)
    forwarded_classes = forwarded.argmax(dim = 2)
    cont_model = []
    cont_input = []
    cont_target = []
    for i,sent in enumerate(forwarded_classes):
        words_model = [id_to_de[idx] for idx in sent]
        if '</s>' in words_model:
            pad_idx = words_model.index('</s>')
            words_model = words_model[:pad_idx]

        words_input = [id_to_en[idx] for idx in en_sample[i]]
        if '</s>' in words_input:
            pad_idx = words_input.index('</s>')
            words_input = words_input[:pad_idx]
            
        words_target = [id_to_de[idx] for idx in de_out_sample[i]]
        if '</s>' in words_target:
            pad_idx = words_target.index('</s>')
            words_target = words_target[:pad_idx]

        with MosesDetokenizer('de') as detokenize_de:
            with MosesDetokenizer('en') as detokenize_en:

                cont_model.append(detokenize_de(words_model))
                cont_input.append(detokenize_en(words_input))
                cont_target.append(detokenize_de(words_target))
    sum = 0
    total = cfg.batch_size
    for i in range(cfg.batch_size):
        print(f'############ {i}th sample ##########')
        print(f'input sent = {cont_input[i]}')
        print(f'target sent = {cont_target[i]}')
        print(f'model output = {cont_model[i]}')
        score = bleu_score.sentence_score(cont_model[i], [cont_target[i]])
        sum += score.score
        print(f'bleu for this pair = {score}')

        print('\n')
    print(f'avg BLEU score for batch = {sum / total}')

def eval_bleu(cfg, model, id_to_en, id_to_de):
    model.eval()
    bleu_score = BLEU()

    val_data = pickle.load(open(cfg.data_folder_path + 'test_indices_tuple', 'rb'))
    zipped = list(zip(val_data[0], val_data[1], val_data[2]))
    total_len = len(zipped)
    batch_iter = int(total_len / cfg.batch_size)
    cont_model = []
    cont_input = []
    cont_target = []
    
    for b in range(batch_iter):
        dat = zipped[b * cfg.batch_size : min((b + 1) * cfg.batch_size, total_len)]
        en_val, de_in_val, de_out_val = zip(*dat)

        en_lst = [torch.tensor(en_line) for en_line in en_val]
        en_sample = nn.utils.rnn.pad_sequence(en_lst,batch_first = True).to(cfg.device)

        de_in_lst = [torch.tensor(de_in_line) for de_in_line in de_in_val]
        de_in_sample = nn.utils.rnn.pad_sequence(de_in_lst,batch_first = True).to(cfg.device)

        de_out_lst = [torch.tensor(de_out_line) for de_out_line in de_out_val]
        de_out_sample = nn.utils.rnn.pad_sequence(de_out_lst,batch_first = True).to(cfg.device)
    
        forwarded = model(en_sample, de_in_sample)
        forwarded_classes = forwarded.argmax(dim = 2)
        for i,sent in enumerate(forwarded_classes):

            words_model = [id_to_de[idx] for idx in sent]
            if '</s>' in words_model:
                pad_idx = words_model.index('</s>')
                words_model = words_model[:pad_idx]

            words_input = [id_to_en[idx] for idx in en_sample[i]]
            if '</s>' in words_input:
                pad_idx = words_input.index('</s>')
                words_input = words_input[:pad_idx]
                
            words_target = [id_to_de[idx] for idx in de_out_sample[i]]
            if '</s>' in words_target:
                pad_idx = words_target.index('</s>')
                words_target = words_target[:pad_idx]
            
            with MosesDetokenizer('de') as detokenize_de:
                with MosesDetokenizer('en') as detokenize_en:

                    cont_model.append(detokenize_de(words_model))
                    # cont_input.append(detokenize_en(words_input))
                    cont_target.append(detokenize_de(words_target))

    sum = 0
    total = len(cont_model)
    for i in range(total):
        
        score = bleu_score.sentence_score(cont_model[i], [cont_target[i]])
        sum += score.score
        # print(f'bleu for this pair = {score}')

        print('\n')
    print(f'avg BLEU score for test data with total {total} pairs = {sum / total}')

    return sum / total
        
    



if __name__ == '__main__':
    cfg = Config()
    '''evaluating model'''
    cfg.batch_size = 128
    # if torch.cuda.is_available:
    if 0:
        device = torch.device('cuda')
        cfg.device = device
    else:
        device = torch.device('cpu')

    print(f'device = {device}')

    checkpoint = torch.load('./results/general_currentmodel', map_location = device)
    model = RNN(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id = lookup_table_maker(cfg)

    # see_translation(cfg, model, id_to_en, id_to_de)
    eval_bleu(cfg, model, id_to_en, id_to_de)


    ######################################################################################################################
    '''checking dataset'''

    # val_data = pickle.load(open('./data/train_indices_tuple', 'rb'))
    # en_vocab, de_vocab, id_to_en, en_to_id, id_to_de, de_to_id = lookup_table_maker(cfg)
    # en_val, de_in_val, de_out_val = val_data

    # for k in range(300):
    #     en_sample = en_val[k]
    #     check_input = [id_to_en[i] for i in en_sample]
    #     print(check_input)

    #     de_in_sample = de_in_val[k]
    #     check_decoder_in = [id_to_de[i] for i in de_in_sample]
    #     print(check_decoder_in)
    #     print('\n')

    ###########################################################################################################################
    '''checking raw file '''
    # with open(cfg.en_train_path) as f:
    #     en_file = f.readlines()
    # with open(cfg.de_train_path) as k:
    #     de_file = k.readlines()

    # for i in range(1000):
    #     print(f'############ {i}th sample ##########')
    #     print(en_file[i])
    #     print(de_file[i])
    #     print('\n')
