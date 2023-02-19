import os
import sys
import warnings
warnings.filterwarnings("error")

sys.path.append(os.getcwd())

from preprocess import *
import numpy as np
from evaluation_temp import *

def train(model_type = cfg.model_type, loss_type = cfg.loss_type,subsampling_t = cfg.subsampling_threshold):
    """필요한 변수, 경로 선언하고 pickle만들기"""
    cfg = Config()
    random.seed(1128)
    hidden_size = cfg.hidden_size
    neg_num = cfg.NEG
    epochs = cfg.epoch
    min_loss = math.inf


    emb_save_path = './results/embedding_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
                                                                      subsampling_t, epochs)
    cont_save_path = './results/context_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
                                                                     subsampling_t, epochs)
    nodevec_mat_save_path = './results/nodevec_mat_{}_{}_{}_{}epoch.pkl'.format(model_type, loss_type,
                                                                          subsampling_t, epochs)

    if not os.path.isfile(cfg.vocabulary_path):

        make_pickles()
    with open(cfg.vocabulary_path, 'rb') as f:
        vocabulary = pickle.load(f)
    with open(cfg.word_to_index_path, 'rb') as f:
        word_to_index = pickle.load(f)
    with open(cfg.index_to_word_path, 'rb') as f:
        index_to_word = pickle.load(f)
    with open(cfg.unigram_table_path, 'rb') as f:
        unigram_table = pickle.load(f)
    if cfg.model_type == 'CBOW':
        with open(cfg.huffman_tree_path, 'rb') as f:
            huffman_tree = pickle.load(f)
        with open(cfg.bin_paths_path, 'rb') as f:
            bin_paths = pickle.load(f)
        with open(cfg.index_container_path, 'rb') as f:
            index_container = pickle.load(f)

    unigram_table = np.array(unigram_table)
    vocab_size = len(vocabulary)
    len_unigram_table = len(unigram_table)
    # total_words = sum([item[1] for item in frequency.items()])
    total_words = sum(vocabulary.values())
    word_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(vocab_size, hidden_size)).astype('f')
    context_emb = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(vocab_size, hidden_size)).astype('f')
    nodevec_mat = np.random.uniform(low=-0.5 / 300, high=0.5 / 300, size=(vocab_size - 1, hidden_size)).astype('f')


    '''training 부분'''
    step = 0
    start_time = time.time()
    lr = cfg.learning_rate
    print(f'Start training on {len(cfg.train_files)} with {total_words} words, {vocab_size} vocab')

    for epoch in range(cfg.epoch):
        print("======= Epoch {} training =======".format(epoch + 1))
        sibal_counter = 0
        for i in range(len(cfg.train_files)):
            file_count = 0
            file_start_time = time.time()
            file = cfg.train_files[i]
            file_start_time =time.time()
            print(f'training on {i + 1}th file : {file}')
            file_in, file_out = preprocess(file, cfg.window_size, word_to_index, vocabulary, index_to_word, total_words, threshold = subsampling_t)

            # print(np.array(file_in))
            loss = 0
            lr_update_count = 0
            lr = cfg.learning_rate * (100 - i) / 100
            # for j in range(len(file_in)):
            for j in tqdm(range(len(file_in)), desc = f"file{i} training on lr :{lr}"):
                if model_type == 'SG':

                    temp_input = file_in[j]
                    # print(temp_input)
                    temp_output = file_out[j]
                    # print(temp_output)
                    ctxt_len = len(temp_output)

                elif model_type == "CBOW":
                    temp_input = file_out[j]
                    temp_output = file_in[j]
                    if len(temp_input) == 0:
                        sibal_counter += 1
                        continue


                if loss_type == 'ng':
                    neg_idx = np.random.randint(low=0, high=len_unigram_table, size=neg_num * len(temp_output))
                    neg_sample = unigram_table[neg_idx]
                    # print(neg_sample)
                    
                    temp_output.extend(neg_sample)
                    # print(temp_output)
                    context = context_emb[temp_output]

                else:    #hs
                    path_arr = path_to_arr(int(bin_paths[temp_output]))
                    node_indices = np.array(index_container[temp_output]) - vocab_size
                    node_mat = nodevec_mat[node_indices]

                if model_type == "SG":
                    hidden = word_emb[np.array(temp_input)]
                    hidden = hidden.reshape(-1,cfg.hidden_size)

                else:       #hs
                    
                    hidden = word_emb[np.array(temp_input)].sum(axis = 0) / len(temp_input)
                    hidden.reshape(-1,cfg.hidden_size)

                if loss_type == 'ng':
                    out = sigmoid(np.dot(hidden, context.T))
                    # if np.any(np.isnan(test1)):s
                    p_loss = -np.log(out[:, :ctxt_len] + 1e-7)
                    n_loss = -np.sum(np.log(1 - out[:, ctxt_len:] + 1e-7))
                    loss += (np.sum(p_loss) + n_loss)

                    out[:, :ctxt_len] -= 1

                    context_grad = np.dot(out.T, hidden)

                    emb_grad = np.dot(out, context_emb[temp_output])

                elif loss_type == 'hs':
                    # print(f'path_arr : {path_arr}')
                    # print(f'out : {np.dot(hidden, node_mat.T)}')
                    out = np.dot(hidden, node_mat.T)
                    out = sigmoid(out)
                    # print(f'sigmoided : {out}')
                    pair_loss = -np.sum(np.log(abs(path_arr - out + 1e-6)))
                    # print(f'pair_loss : {np.log(out + 1e-6)}')
                    loss += pair_loss
                    # dout = path_arr + out - 1  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (1, depth)
                    
                    dout = path_arr + out - 1
                    # print(f'hidden shape : {hidden.shape}')
                    node_mat_grad = np.dot(dout.reshape(len(dout),-1), hidden.reshape(-1,300))  # (depth, 300)
                    # print(f'node_mat_grad shape : {node_mat_grad.shape}')
                    # print(f'node_mat.T.shape : {node_mat.T.shape}')
                    emb_grad = np.dot(dout.reshape(-1,len(dout)), node_mat)  # (1, 300)
                    # print(emb_grad.shape)
                    

                # print(context_grad)
                if model_type == 'CBOW':
                    emb_grad /= len(temp_input)
                    nodevec_mat[node_indices] -= lr * node_mat_grad
                if model_type == 'SG':
                    for k, target in enumerate(temp_output):
                        context_emb[target] -= lr * context_grad[k]
                # context_emb[temp_output] -= lr * context_grad
                # word_emb[temp_input] -= lr * emb_grad.squeeze()
                word_emb[temp_input] -= lr * emb_grad
                file_count += 1

            print("Number of pairs trained in this file: {}".format(file_count))
            print("Loss: {:.5f}".format(loss / file_count))
            print(f'sibal_counter = {sibal_counter}')
            print("Took {:.2f} hours for single file".format((time.time() - file_start_time) / 3600))

            if loss < min_loss:
                min_loss = loss
                pickle.dump(word_emb, open(emb_save_path, 'wb'))
                if loss_type == 'ng':
                    pickle.dump(context_emb, open(cont_save_path, 'wb'))
                else:
                    pickle.dump(nodevec_mat, open(nodevec_mat_save_path, 'wb'))
            # similar_word(embedding)
            # monitor = eval_monitor(word_emb, vocabulary, word_to_index, eval_point = i, k = 5) #eval_point에는 현재 step(file) 넣기 (저장용)
            
    print("Training time: {:.2f} hours".format((time.time() - start_time) / 3600))



def sigmoid(x):
	l=len(x)
	y=[]
	for i in range(l):
		if x[i]>=0:
			y.append(1.0/(1+np.exp(-x[i])))
		else:
			y.append(np.exp(x[i])/(np.exp(x[i])+1))
	return np.array(y)




train()








