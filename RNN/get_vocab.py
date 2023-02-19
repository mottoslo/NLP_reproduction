from urllib.request import urlretrieve

URL = {
    "en_train": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en",
    "de_train": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de",
    "en_val": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en",
    "de_val": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de",
    "en_test2014": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en",
    "de_test2014": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de",
    "en_test2015": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en",
    "de_test2015": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de",
    "en_vocab": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en",
    "de_vocab": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de",
    "en-de_dict": "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de"
}

for filename, url in URL.items():
    urlretrieve(url, filename)