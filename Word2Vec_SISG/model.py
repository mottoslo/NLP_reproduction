# from config import *
# from utils import *
# from tqdm.auto import tqdm
# import numpy as np
# import time
# import pickle
# import os
# import math

# class SISG():
#     def __init__(self,cfg):
#         print('loading pickles.....')
#         self.vocabulary = pickle.load(open(cfg.pickle_path + '/vocabulary', 'rb'))
#         # self.word_to_index = pickle.load(open(cfg.pickle_path + './word_to_index.pkl', 'rb'))
#         # self.index_to_word = pickle.load(open(cfg.pickle_path + './index_to_word.pkl', 'rb'))
        
#         self.ngram_vocabulary = pickle.load(open(cfg.pickle_path + '/ngram_vocabulary', 'rb'))
#         self.ngram_word_to_index = pickle.load(open(cfg.pickle_path + '/ngram_word_to_index', 'rb'))
#         self.ngram_index_to_word = pickle.load(open(cfg.pickle_path + '/ngram_index_to_word', 'rb'))
#         self.unigram_table = np.array(pickle.load(open(cfg.pickle_path + '/unigram_table', 'rb')))
#         self.word_to_ngram_indices = pickle.load(open(cfg.pickle_path + '/word_to_ngram_indices', 'rb'))
#         print('pickle loaded!')

#         self.ngram_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(len(self.ngram_vocabulary), cfg.emb_dim)).astype('f')
#         self.context_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(len(self.ngram_vocabulary), cfg.emb_dim)).astype('f')
        
                                                            
#         self.ngram_emb_save_path = f'./results/ngram_embedding.pkl'
#         self.context_emb_save_path = f'./results/context_embedding.pkl'

#         self.len_unigram_table = len(self.unigram_table)
#         self.total_words = sum(self.vocabulary.values())
#         self.num_neg_each = cfg.num_neg_each
#         self.min_loss = math.inf
#         self.num_epoch = cfg.num_epoch
    
#     def get_emb_rep(self, ngram_indices):
#         rep = self.ngram_emb[ngram_indices].sum(axis = 0) / len(ngram_indices)
#         return rep

#     def get_context_rep(self, ngram_indices):
#         rep = self.context_emb[ngram_indices].sum(axis = 0) / len(ngram_indices)
#         if len(ngram_indices) == 0:
#             quit()
#         return rep
        
#     def train_step(self, center_word_idx, context_words_idx, lr, ctxt_len):  # ctxt_len은 positive sample의 개수
#         # print(f'center word = {center_word_idx}')
#         # print(f'context word = {context_words_idx}')
#         center_word_ngram_indices = self.word_to_ngram_indices[center_word_idx]
#         center_word_rep = self.get_emb_rep(center_word_ngram_indices).reshape(-1, cfg.emb_dim)
#         # center_word_rep = self.get_emb_rep(center_word_ngram_indices)
#         # print(f'context_words_idx = {context_words_idx}')
#         context_word_rep = np.array([self.get_context_rep(self.word_to_ngram_indices[index]) for index in context_words_idx])
#         # print(f' center_rep = {center_word_rep}')
#         # print(f' context_rep = {context_word_rep}')
#         # print(f' ctxt_len = {ctxt_len}')
#         # print(f' dotted = {np.dot(center_word_rep, context_word_rep.T)}')
#         # print(f' dotted shape = {np.dot(center_word_rep, context_word_rep.T).shape}')

#         out = sigmoid(np.dot(center_word_rep, context_word_rep.T)[0])
#         out = out.reshape(len(out),-1)
#         # out = out.reshape(-1,len(out))
#         # if np.any(np.isnan(test1)):
#         # p_loss = -np.log(out[:, :ctxt_len] + 1e-7)
#         # n_loss = -np.sum(np.log(1 - out[:, ctxt_len:] + 1e-7))
        
#         p_loss = -np.log(out[:ctxt_len] + 1e-7)
#         n_loss = -np.sum(np.log(1 - out[ctxt_len:] + 1e-7))
#         # print(f'p loss = {p_loss} // n loss = {n_loss}')
#         loss = (np.sum(p_loss) + n_loss)

#         out[:ctxt_len] -= 1
#         # print(f' out shape = {out.shape}')

#         context_grad = np.dot(out, center_word_rep)
#         # print(f' context_grad shape = {context_grad}')

#         emb_grad = np.dot(out.T, context_word_rep)
#         # print(f' emb_grad shape = {emb_grad}')


#         self.ngram_emb[center_word_ngram_indices] -= lr * emb_grad

#         for i, word in enumerate(context_words_idx):
#             self.context_emb[self.word_to_ngram_indices[word]] -= lr * context_grad[i]
#             # if np.any(np.isnan(context_grad)):
#             #     print(f'i = {i}')
#             #     print(f'center_word_idx = {center_word_idx}')
#             #     print(f'center_word_idx = {context_words_idx}')
#             #     print(f'123123 = {context_grad[i]}')
#             #     quit()
        
#         return loss
    
#     def train(self):
#         step_size = 0
#         lr = cfg.starting_lr
#         sibal_counter = 0
#         sibal2_counter = 0
#         for j in range(50):
#             file_count = 0
#             loss = 0
#             file_start_time = time.time()
#             print(f'loading {j}th file....')
#             center_words = pickle.load(open(f'./data/split/wiki_center_words_{j}.pkl', 'rb'))
#             context_words = pickle.load(open(f'./data/split/wiki_context_words_{j}.pkl', 'rb'))
#             if len(center_words) != len(context_words):
#                 print(f'center_word, context_word length mismatch in file{j}')
#                 quit()

#             for i, context_word_idx in tqdm(enumerate(context_words), desc = f'training on file_{j}', ncols =50):
#                 if center_words[i] == 2:
#                     # print('sibal')
#                     sibal_counter += 1
#                     continue

#                 context_word_idx = remove_two(context_word_idx)
                
#                 ctxt_len = len(context_word_idx)
#                 neg_idx = np.random.randint(low=0, high=self.len_unigram_table, size = cfg.num_neg_each * ctxt_len)
#                 neg_sample = self.unigram_table[neg_idx]
        
#                 context_word_idx.extend(neg_sample)
#                 context_word_idx = remove_two(context_word_idx)
#                 if len(context_word_idx) == 0:
#                     # print('sibal2')
#                     sibal2_counter +=1
#                     continue

#                 lr = cfg.starting_lr * (1 - (step_size / (cfg.num_epoch * self.total_words)))
#                 loss += self.train_step(center_words[i], context_word_idx, lr = lr, ctxt_len = ctxt_len)

#                 step_size += 1
#                 file_count +=1

#             print("Number of pairs trained in this file: {}".format(file_count))
#             print("Loss: {:.5f}".format(loss / file_count))
#             print(f'sibal_counter = {sibal_counter} // sibal2_counter = {sibal2_counter}')
#             print("Took {:.2f} hours for single file".format((time.time() - file_start_time) / 3600))
#             pickle.dump(self.ngram_emb, open(f'./results/ngram_emb_after_file{j}', 'wb'))



# cfg = Config()
# model = SISG(cfg)
# model.train()
        
        
        
        

        


    
    