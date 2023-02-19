from preprocess import *
import config

cfg = Config()

if __name__ == '__main__':
    vocabulary = pickle.load(open(cfg.vocabulary_path, 'rb'))
    index_to_word = pickle.load(open(cfg.index_to_word_path, 'rb'))
    word_to_index = pickle.load(open(cfg.word_to_index_path, 'rb'))
    emb = pickle.load(open('./results/final/skip-neg/embedding_SG_ng_1e-05_1epoch.pkl', 'rb'))

    for i in range(10):
        print(index_to_word[i])
    

    def index_to_path(idx):
        bnr = bin(idx). replace('0b','')
        bnr = np.array([int(i) for i in bnr[1:]])

        node_mat_idx = [0]
        idx_count = 1
        for direction in bnr:
            if direction:
                idx_count = idx_count * 2 + 1
            else:
                idx_count = idx_count * 2
            node_mat_idx.append(idx_count - 1)
        node_mat_idx = np.array(node_mat_idx)
        return bnr, node_mat_idx

    path, idx = index_to_path(15)
    print(path)
    print(idx)