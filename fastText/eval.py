import numpy as np
import pickle
from tqdm.auto import tqdm
from preprocess import *
from evaluation_temp import *

def evaluation(cfg, task):
    eval_txt = pickle.load(open(cfg.test_data_save_path + f'/{task}_text_test', 'rb'))
    eval_label = pickle.load(open(cfg.test_data_save_path + f'/{task}_label_test', 'rb'))
    word_emb = pickle.load(open(cfg.result_emb_save_path + f'/word_emb_{task}', 'rb'))
    
    
    