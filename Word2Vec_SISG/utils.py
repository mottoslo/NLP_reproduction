
import numpy as np
import math


def readWord(word, n_gram_num):
    cont = []
    new_word = '<' + word + '>'
    for c in range(len(new_word) - n_gram_num + 1):
        cont.append(new_word[c : c + n_gram_num])

    return cont

def readWord_all_ngrams(word, n_gram_min = 3, n_gram_max = 6):
    cont = []
    for i in range(n_gram_min, n_gram_max + 1):
        cont.extend(readWord(word, i))
    if (len(word) + 2) > n_gram_max:
        cont.append('<' + word + '>')
    
    return cont
    
    

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

def subsampling(word_idx, vocabulary, index_to_word, total, threshold):
    """논문 subsampling 식에 의거하여 word를 포함하면 True, 제외하면 False를 리턴"""
    freq_ratio = vocabulary[index_to_word[word_idx]] / total
    discard_prob = 1 - np.sqrt((threshold / freq_ratio))
    rand = np.random.uniform()
    decision = True if rand > discard_prob else False
    word = index_to_word[word_idx]
    # print(f'word : {word} // freq_ratio : {freq_ratio} // discard_prob : {discard_prob} // draw : {rand} // decision : {decision}')

    return decision

def sigmoid(x):
	l=len(x)
	y=[]
	for i in range(l):
		if x[i]>=0:
			y.append(1.0/(1+np.exp(-x[i])))
		else:
			y.append(np.exp(x[i])/(np.exp(x[i])+1))
	return np.array(y)
        
def remove_two(x):
    x = [i for i in x if i != 2]

    return x

def get_emb_rep(self, ngram_indices):
    rep = self.ngram_emb[ngram_indices].sum(axis = 0) / len(ngram_indices)
    return rep

def get_context_rep(self, ngram_indices):
    rep = self.context_emb[ngram_indices].sum(axis = 0) / len(ngram_indices)
    if len(ngram_indices) == 0:
        quit()
    return rep

    
