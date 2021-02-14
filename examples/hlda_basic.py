import sys
import tomotopy as tp
import numpy as np

def hlda_example(input_file, save_path):
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    try:
        cps = tp.utils.Corpus.load(input_file + '.cached.cps')
    except IOError:
        stemmer = PorterStemmer()
        stops = set(stopwords.words('english'))
        cps = tp.utils.Corpus(
            tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
            stopwords=lambda x: len(x) <= 2 or x in stops
        )
        cps.process(open(input_file, encoding='utf-8'))
        cps.save(input_file + '.cached.cps')
    
    np.random.seed(42)
    ridcs = np.random.permutation(len(cps))
    test_idcs = ridcs[:20]
    train_idcs = ridcs[20:]

    test_cps = cps[test_idcs]
    train_cps = cps[train_idcs]
    
    mdl = tp.HLDAModel(tw=tp.TermWeight.ONE, min_df=10, depth=4, rm_top=10, corpus=train_cps)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for _ in range(0, 1000, 10):
        mdl.train(7)
        mdl.train(3, freeze_topics=True)
        print('Iteration: {:05}\tll per word: {:.5f}\tNum. of topics: {}'.format(mdl.global_step, mdl.ll_per_word, mdl.live_k))

    for _ in range(0, 100, 10):
        mdl.train(10, freeze_topics=True)
        print('Iteration: {:05}\tll per word: {:.5f}\tNum. of topics: {}'.format(mdl.global_step, mdl.ll_per_word, mdl.live_k))

    mdl.summary()
    print('Saving...', file=sys.stderr, flush=True)
    mdl.save(save_path, True)

    test_result_cps, ll = mdl.infer(test_cps)
    for doc in test_result_cps:
        print(doc.path, doc.get_words(top_n=10))

# You can get the sample data file 'enwiki-16000.txt'
# at https://drive.google.com/file/d/1OfyJ9TqaMiqzO6Qw-c_jXL-pmSIZf5Xt/view?usp=sharing

if __name__ == '__main__':
    hlda_example('enwiki-16000.txt', 'test.hlda.tmm')
