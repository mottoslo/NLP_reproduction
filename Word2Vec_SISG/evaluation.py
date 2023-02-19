import numpy as np
import pickle
from tqdm.auto import tqdm
from preprocess import *
import config

def make_evaluation_tokens(word_to_index):
    category_names = []
    tokens_array = []
    cat_tokens = []

    with open(cfg.evaluation_data) as f:
        lines = f.readlines()

        for line in lines:
            if line[0] == ":":
                if len(cat_tokens) != 0:    
                    tokens_array.append(cat_tokens)
                category_names.append(line[2:-1])
                cat_tokens = []

            else:
                line = clean_str(line)
                tokens = tokenize(line)
                temp = 0
                for token in tokens:
                    if token not in word_to_index.keys():
                        continue
                    else:
                        temp += 1
                if temp == 4:
                    cat_tokens.append(tokens)
        if len(cat_tokens) != 0:
            tokens_array.append(cat_tokens)
        
    for i in range(len(category_names)):

        pickle.dump(tokens_array[i], open(f'./data/eval_data/questions-answers/{category_names[i]}', 'wb'))
        print('pickle dumped')

def cosine_similarity_matrix(x,y, normalized = False):   #x는 matrix, y는 벡터하나
    if not normalized:
        x_norm = np.linalg.norm(x,axis = 1)
        x = x / x_norm[:, None]
        y = y / np.linalg.norm(y)
    
    sim = np.dot(x,y.transpose())

    return sim
def cosine_similarity(x,y, normalized = False):
    if not normalized:
        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
    sim = np.dot(x,y)

    return sim

def eval_monitor(emb, vocabulary, word_to_index, eval_point, k = 10):
    categories = os.listdir('./data/eval_data/questions-answers')
    embedding_norm = np.linalg.norm(emb, axis=1)
    emb = emb / embedding_norm[:, None]
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    
    result_dict = {}
    for filename in categories:
        tokens = pickle.load(open(f'./data/eval_data/questions-answers/{filename}', 'rb'))
        cat_correct_percentage = 0
        sample_len = 0
        correct = 0 
        rand_idx =  random.sample(range(len(tokens)),100)
        tokens = np.array(tokens)
        for sample in tqdm(tokens[rand_idx], desc = f'evaluating sample on {filename}'):
            topk = find_nearest_k(emb, sample[:3], word_to_index, normalized = True, k = k)

            if word_to_index[sample[3]] in topk:
                correct += 1
                print(f'correct word : {sample[3]}')
                print(f'input words : {sample}')
                print(f'topk list')
                for i in topk:
                    print(index_to_word[i])
            sample_len += 1
        cat_correct_percentage = correct / sample_len
        result_dict[filename] = cat_correct_percentage
        print(f'correct word in top {k} on {filename} : {cat_correct_percentage * 100}')
    pickle.dump(result_dict, open(f'./results/accuracy_per_category_{eval_point}','wb'))
    print(f'avg % : {sum(result_dict.values()) / len(categories) * 100}')
    return sum(result_dict.values()) / len(categories)
    

def similar_word_temp(norm_emb, normalized = True): #normalized emb
    if normalized == False:
        embedding_norm = np.linalg.norm(emb, axis=1)
        norm_emb = norm_emb / embedding_norm[:, None]

    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))

    word1 = word_to_index['man']
    word2 = word_to_index['king']
    word3 = word_to_index['woman']
    answer = word_to_index['queen']


    target = norm_emb[word2] - norm_emb[word1] + norm_emb[word3]
    target = target / np.linalg.norm(target)

    max_index = answer
    max_sim = cosine_similarity(target, norm_emb[answer])
    for i in tqdm(range(len(word_to_index)), desc="Finding closest word to king - man + woman", ncols=70):
        if i == word1 or i == word2 or i == word3 or i == answer:
            pass
        else:
            sim = cosine_similarity(norm_emb[i], target)
            if sim > max_sim:
                max_sim = sim
                max_index = i
    print(index_to_word[max_index])

def find_nearest_k(emb,input_tokens, word_to_index, normalized = False, k = 5):
    """return nearest_k index in list"""
    if normalized == False:
        embedding_norm = np.linalg.norm(emb, axis=1)
        emb = emb / embedding_norm[:, None]
    input_indices = [word_to_index[i] for i in input_tokens]
    start_time = time.time()
    input_vec = emb[input_indices]
    query_vec = input_vec[1] - input_vec[0] + input_vec[2]
    sim_scores = cosine_similarity_matrix(emb, query_vec, normalized = True)
    order = sim_scores.argsort()
    flipped = np.flip(order)
    topk_index = flipped[:k + 4]
    topk_index = [i for i in flipped if i not in input_indices]

    return topk_index[:k]

def eval_overall(emb, vocabulary, word_to_index, index_to_word, eval_point,k = 5):
    categories = os.listdir('./data/eval_data/questions-answers')
    embedding_norm = np.linalg.norm(emb, axis=1)
    emb = emb / embedding_norm[:, None]
    
    result_dict = {}
    total_sample_count = 0
    total_correct_count = 0
    for filename in categories:
        tokens = pickle.load(open(f'./data/eval_data/questions-answers/{filename}', 'rb'))
        cat_correct_percentage = 0
        sample_len = 0
        correct = 0 
        for sample in tqdm(tokens, desc = f'evaluating on {filename}'):
            topk = find_nearest_k(emb, sample[:3], word_to_index, normalized = True, k = k)

            if word_to_index[sample[3]] in topk:
                correct += 1
                # print(f'correct word : {sample[3]}')
                # print(f'input words : {sample}')
                # print(f'topk list')
                # for i in topk:
                #     print(index_to_word[i])
            sample_len += 1
        total_sample_count += sample_len
        total_correct_count += correct
        cat_correct_percentage = correct / sample_len
        result_dict[filename] = cat_correct_percentage
        print(f'correct word % in {filename} : {cat_correct_percentage * 100}')
    pickle.dump(result_dict, open(f'./results/final/accuracy_per_category_{eval_point}','wb'))
    # return sum(result_dict.values()) / len(categories) * 100
    return (total_correct_count / total_sample_count) * 100

def compositionality(emb, vocabulary, word_to_index, index_to_word, eval_point = 1000, k = 10):
    embedding_norm = np.linalg.norm(emb, axis=1)
    emb = emb / embedding_norm[:, None]
    q = [["czech", "currency"], ["vietnam", "capital"], ["german","airlines"], ["russian","river"], ["french","actress"],["korean","food"]]
    near_containter = []
    for pair in q:
        query_vec = emb[word_to_index[pair[0]]] + emb[word_to_index[pair[1]]]
        sim_scores = cosine_similarity_matrix(emb, query_vec, normalized = True)
        order = sim_scores.argsort()
        flipped = np.flip(order)
        topk_index = flipped[2:k]
        print(f'Query : {pair}')
        for idx in topk_index:
            print(index_to_word[idx])
    



if __name__ == '__main__':
    cfg = config.Config()

    # categories = os.listdir('./data/eval_data/questions-answers')
    # for filename in categories:
    #     temp = pickle.load(open(f'data/eval_data/questions-answers/{filename}', 'rb'))
    #     print(temp)
    #     quit()


    vocabulary = pickle.load(open('./pickles/ngram_vocabulary', 'rb'))
    index_to_word = pickle.load(open('./pickles/index_to_word', 'rb'))
    word_to_index = pickle.load(open('./pickles/word_to_index', 'rb'))
    ngram_word_to_index = pickle.load(open('./pickles/ngram_word_to_index' , 'rb'))
    emb = pickle.load(open('./results/last_ngram_emb', 'rb'))
    make_word_to_ngram_indices(cfg, vocabulary, word_to_index, ngram_word_to_index)
