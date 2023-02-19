import numpy as np
from utils import * #softmax 짜기
import config
import pickle
import math
import time
import random


class fastText():
    def __init__(self, cfg):

        self.vocabulary = pickle.load(open(cfg.pickle_save_path + f'/vocabulary_{cfg.data_type}.pkl', 'rb'))
        self.bigram_vocabulary = pickle.load(open(cfg.pickle_save_path + f'/bigram_vocabulary_{cfg.data_type}.pkl', 'rb'))

        self.word_to_index = pickle.load(open(cfg.pickle_save_path + f'/word_to_index_{cfg.data_type}.pkl', 'rb'))
        self.bigram_word_to_index = pickle.load(open(cfg.pickle_save_path + f'/bigram_word_to_index_{cfg.data_type}.pkl', 'rb'))

        self.index_to_word = pickle.load(open(cfg.pickle_save_path + f'/index_to_word_{cfg.data_type}.pkl', 'rb'))
        self.bigram_index_to_word = pickle.load(open(cfg.pickle_save_path + f'/bigram_index_to_word_{cfg.data_type}.pkl', 'rb'))


        self.word_emb = np.random.uniform(low=-0.5 / 10, high=0.5 / 10, size=(len(self.vocabulary) + 1, cfg.hidden_size)).astype('f')
        self.bigram_emb = np.random.uniform(low=-0.5 / 10, high=0.5 / 10, size=(len(self.bigram_vocabulary) + 1, cfg.hidden_size)).astype('f')
        
        self.fc1 = np.random.uniform(low = 0.5 / 10, high = 0.5 / 10, size=(cfg.hidden_size, cfg.num_classes[cfg.data_type])).astype('f')
        self.task = cfg.data_type
        self.best_emb = self.word_emb = np.random.uniform(low=-0.5 / 10, high=0.5 / 10, size=(len(self.vocabulary) + 1, cfg.hidden_size)).astype('f')


    def train_step(self, tokens, y, lr):
        if cfg.use_bigram == True:
            bigram_idx = bigram_tokens_to_indices(tokens, self.bigram_word_to_index, self.bigram_vocabulary)
            bigram_len = len(bigram_idx)

        single_idx = tokens_to_indices(tokens, self.word_to_index, self.vocabulary)
        single_len = len(single_idx)

        single_rep = self.word_emb[single_idx].sum(axis = 0) / single_len

        if cfg.use_bigram == True:
            bigram_rep = self.bigram_emb[bigram_idx].sum(axis = 0) / bigram_len
            hidden = (single_rep + bigram_rep) / 2
        else:
            hidden = single_rep / single_len
        hidden = hidden.reshape(-1, cfg.hidden_size)
        
        out = np.dot(hidden, self.fc1)    
        
        softmaxed = softmax(out)
        
        loss = -np.log(softmaxed[0,y - 1])
        
        softmaxed[0, y-1] -= 1

        W_grad = np.dot(hidden.T, softmaxed)
        emb_grad = np.dot(softmaxed, self.fc1.T)
        if cfg.use_bigram == True:
            emb_grad = emb_grad / 2
        self.fc1 -= lr * W_grad
        self.word_emb[single_idx] -= lr * (emb_grad / single_len)
        if cfg.use_bigram == True:

            self.bigram_emb[bigram_idx] -= lr * (emb_grad / bigram_len)
        
        return loss               
    
    def train(self,cfg):
        train_start_time = time.time()
        train_label = pickle.load(open(cfg.train_data_save_path + f'/{cfg.data_type}_label_train.pkl', 'rb'))
        train_dat = pickle.load(open(cfg.train_data_save_path + f'/{cfg.data_type}_text_train.pkl', 'rb'))

        shuffle_start_time = time.time()
        zipped = list(zip(train_dat,train_label))
        random.shuffle(zipped)
        dev_zipped = zipped[:int(len(zipped) / 15)]
        zipped = zipped[int(len(zipped) / 15) :]

        
        train_dat, train_label = zip(*zipped)
        dev_dat, dev_label = zip(*dev_zipped)
        print(f'data shuffling time taken : {time.time() - shuffle_start_time}')

        lr = cfg.starting_lr
        total_sent = len(train_dat)
        loss_track = 0
        sibal_counter = 0
        sibal_2_counter = 0
        lr_step = 1
        for epoch in range(cfg.num_epoch):
            
            shuffle_start_time = time.time()
            zipped = list(zip(train_dat,train_label))
            random.shuffle(zipped)
            train_dat, train_label = zip(*zipped)
            print(f'data shuffling time taken : {time.time() - shuffle_start_time}')

            step = 0
            loss = 0
            for i, sent in tqdm(enumerate(train_dat), desc = f'epoch :{epoch + 1} data : {self.task} / starting_lr : {cfg.starting_lr}'):
                sent = tokenize(sent)
                if len(sent) < 2:
                    sibal_counter +=1
                    continue
                if len(sent) > 1000:
                    sibal_2_counter +=1
                    continue

                loss += self.train_step(tokens = sent, y = train_label[i], lr = lr)
                step +=1
                if step == int(len(train_dat) / 100):
                    dev_acc = self.evaluation_dev(dev_dat, dev_label, type = 'current')
                    print(f'train loss : {loss / step} // dev_acc = {dev_acc}',)
                    lr = cfg.starting_lr * ((100 - (lr_step / cfg.num_epoch)) / 100)
                    lr_step += 1


                    if (dev_acc) > loss_track:
                        self.best_emb = self.word_emb
                        loss_track = dev_acc
                    step = 0
                    loss = 0
            self.evaluation(type = 'best')
        print(f'for {cfg.num_epoch} epochs, lr : {cfg.starting_lr} time taken : \
        {time.time() - train_start_time} sibal : {sibal_counter} sibal_2 : {sibal_2_counter}')


    
    def eval_step(self, x, y, type = 'best'):

        tokens = tokenize(x)
        single_idx = tokens_to_indices(tokens, self.word_to_index, self.vocabulary)
        # bigram_idx = bigram_tokens_to_indices(tokens, self.bigram_word_to_index, self.bigram_vocabulary)
        if type == 'best':
            rep = self.best_emb[single_idx].sum(axis = 0)   #저장된 best embedding
        elif type == 'current':
            rep = self.word_emb[single_idx].sum(axis = 0) 
        # bigram_rep = self.bigram_emb[bigram_idx].sum(axis = 0)
        # hidden = (rep + bigram_rep) / (len(single_idx) + len(bigram_idx))
        hidden = rep / len(single_idx)
        hidden = hidden.reshape(-1, cfg.hidden_size)

        out = np.dot(hidden, self.fc1)
        
        softmaxed = softmax(out)
        if softmaxed.argmax() == (y - 1):
            return 1
        else:
            return 0
    
    def evaluation(self, type = 'best'):
        test_label = pickle.load(open(cfg.test_data_save_path + f'/{self.task}_label_test.pkl', 'rb'))
        test_dat = pickle.load(open(cfg.test_data_save_path + f'/{self.task}_text_test.pkl', 'rb'))
        true = 0
        num_data = 0
        for i, sent in tqdm(enumerate(test_dat)):
            true += self.eval_step(x = sent, y = test_label[i],type = type)
            num_data += 1
        acc = (true / num_data) * 100
        print(f'accuracy on {self.task} : {acc}% (type = {type})')
        
        return acc
    
    def evaluation_dev(self, dev_dat, dev_label, type = 'current'):
        test_dat = dev_dat
        test_label = dev_label
        true = 0
        num_data = 0
        for i, sent in tqdm(enumerate(test_dat)):
            true += self.eval_step(x = sent, y = test_label[i], type = type)
            num_data += 1
        acc = (true / num_data) * 100
        print(f'accuracy on {self.task} : {acc}% (type = {type})')
        
        return acc

            

                
        


if __name__ == "__main__":
    cfg = config.Config()
    model = fastText(cfg)
    model.train(cfg)
    model.evaluation(type = 'best')
    model.evaluation(type = 'current')
    model.bigram_emb = 0
    pickle.dump(model, open(cfg.model_save_path + f'/{cfg.data_type}_model', 'wb'))
    
    # train_label = pickle.load(open(cfg.train_data_save_path + f'/{cfg.data_type}_label_train.pkl', 'rb'))
    # train_dat = pickle.load(open(cfg.train_data_save_path + f'/{cfg.data_type}_text_train.pkl', 'rb'))

    # print(model.train_step(x = train_dat[0], y = train_label[0], lr = 0.05))
