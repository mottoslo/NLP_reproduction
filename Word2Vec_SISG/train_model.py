
from config import *
from utils import *
from tqdm.auto import tqdm
import numpy as np
import time
import pickle
import os
import math

def train(cfg):
    print('loading pickles.....')
    vocabulary = pickle.load(open(cfg.pickle_path + '/vocabulary', 'rb'))
    # self.word_to_index = pickle.load(open(cfg.pickle_path + './word_to_index.pkl', 'rb'))
    # self.index_to_word = pickle.load(open(cfg.pickle_path + './index_to_word.pkl', 'rb'))
    
    ngram_vocabulary = pickle.load(open(cfg.pickle_path + '/ngram_vocabulary', 'rb'))
    ngram_word_to_index = pickle.load(open(cfg.pickle_path + '/ngram_word_to_index', 'rb'))
    ngram_index_to_word = pickle.load(open(cfg.pickle_path + '/ngram_index_to_word', 'rb'))
    unigram_table = np.array(pickle.load(open(cfg.pickle_path + '/unigram_table', 'rb')))
    word_to_ngram_indices = pickle.load(open(cfg.pickle_path + '/word_to_ngram_indices', 'rb'))
    print('pickle loaded!')

    ngram_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(len(ngram_vocabulary), cfg.emb_dim)).astype('f')
    context_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(len(ngram_vocabulary), cfg.emb_dim)).astype('f')
                                                        
    ngram_emb_save_path = f'./results/ngram_embedding.pkl'
    context_emb_save_path = f'./results/context_embedding.pkl'

    # len_unigram_table = len(unigram_table)
    # total_words = sum(vocabulary.values())
    # vocab_size = len(vocabulary)

    len_unigram_table = 70231378
    total_words = 140694930
    vocab_size = 1151217

    num_neg_each = cfg.num_neg_each
    min_loss = math.inf
    num_epoch = cfg.num_epoch

    global_step = 0
    start_time = time.time()
    lr = cfg.starting_lr
    print(f'Start training on {50} with {total_words} words, {vocab_size} vocab')

    for epoch in range(cfg.num_epoch):
        print("======= Epoch {} training =======".format(epoch + 1))
        sibal1_counter = 0
        sibal2_counter = 0
        for i in range(50):
            file_count = 0
            file_start_time = time.time()
            file_start_time =time.time()
            print(f'training on {i + 1}th file')
            center_words = pickle.load(open(f'./data/split/wiki_center_words_{i}.pkl', 'rb'))
            context_words = pickle.load(open(f'./data/split/wiki_context_words_{i}.pkl', 'rb'))

            # print(np.array(file_in))
            loss = 0
            lr_update_count = 0
            lr = cfg.starting_lr * (100 - (i * 2)) / 100
            # for j in range(len(file_in)):
            for j in tqdm(range(len(center_words)), desc = f"file{i} training on lr :{lr}"):

                temp_input = center_words[j]
                if temp_input == 2:
                    sibal1_counter +=1
                    continue
                # print(temp_input)
                temp_output = remove_two(context_words[j])
                # print(temp_output)
                ctxt_len = len(temp_output)


                
                neg_idx = np.random.randint(low=0, high=len_unigram_table, size=num_neg_each * ctxt_len)
                neg_sample = unigram_table[neg_idx]
                # print(neg_sample)
                
                temp_output.extend(neg_sample)
                temp_output = remove_two(temp_output)
                if len(temp_output) == 0:
                    sibal2_counter +=1
                    continue
                # print(temp_output)
                
                context = []
                for ngrm in temp_output:
                    ctxt_indices = word_to_ngram_indices[ngrm]
                    rep = (context_emb[ctxt_indices].sum(axis = 0) / len(ctxt_indices))
                    context.append(rep)
                context = np.array(context)

                    

                hidden_indices = word_to_ngram_indices[temp_input]
                hidden = (ngram_emb[hidden_indices].sum(axis = 0) / len(hidden_indices)).reshape(-1, cfg.emb_dim)
                
                out = sigmoid(np.dot(hidden, context.T)[0])
                out = out.reshape(len(out),-1)
                # out = out.reshape(-1,len(out))
                
                p_loss = -np.log(out[:ctxt_len] + 1e-7)
                n_loss = -np.sum(np.log(1 - out[ctxt_len:] + 1e-7))

                loss += (np.sum(p_loss) + n_loss)

                out[:ctxt_len] -= 1

                context_grad = np.dot(out, hidden)

                emb_grad = np.dot(out.T, context)



                # print(context_grad)
                for k, target in enumerate(temp_output):
                    targets = word_to_ngram_indices[target]
                    context_emb[targets] -= lr * context_grad[k]
                # context_emb[temp_output] -= lr * context_grad
                # word_emb[temp_input] -= lr * emb_grad.squeeze()
                ngram_emb[hidden_indices] -= lr * emb_grad
                file_count += 1
                global_step += 1

            print("Number of pairs trained in this file: {}".format(file_count))
            print("Loss: {:.5f}".format(loss / file_count))
            print(f'sibal1_counter = {sibal1_counter} // sibal2_counter = {sibal2_counter}')
            print("Took {:.2f} hours for single file".format((time.time() - file_start_time) / 3600))
            pickle.dump(ngram_emb, open(f'./results/last_ngram_emb', 'wb'), protocol=4)

            # if loss < min_loss:
                # min_loss = loss
                # pickle.dump(ngram_emb, open(f'./results/ngram_emb_after_file{i}', 'wb'))
            # similar_word(embedding)
            # monitor = eval_monitor(word_emb, vocabulary, word_to_index, eval_point = i, k = 5) #eval_point에는 현재 step(file) 넣기 (저장용)
            
    print("Training time: {:.2f} hours".format((time.time() - start_time) / 3600))


cfg = Config()
train(cfg)
    