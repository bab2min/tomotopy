import sys
import tomotopy as tp

def lda_example(input_file, save_path):
    mdl = tp.LDAModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=5, k=20)
    for n, line in enumerate(open(input_file, encoding='utf-8')):
        ch = line.strip().split()
        mdl.add_doc(ch)
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    print('Saving...', file=sys.stderr, flush=True)
    mdl.save(save_path, True)

    for k in range(mdl.k):
        print('Topic #{}'.format(k))
        for word, prob in mdl.get_topic_words(k):
            print('\t', word, prob, sep='\t')


def hdp_example(input_file, save_path):
    mdl = tp.HDPModel(tw=tp.TermWeight.ONE, min_cf=3, rm_top=5)
    for n, line in enumerate(open(input_file, encoding='utf-8')):
        ch = line.strip().split()
        mdl.add_doc(ch)
    mdl.burn_in = 100
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    print('Training...', file=sys.stderr, flush=True)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}\tNum. of topics: {}'.format(i, mdl.ll_per_word, mdl.live_k))

    print('Saving...', file=sys.stderr, flush=True)
    mdl.save(save_path, True)

    important_topics = [k for k, v in sorted(enumerate(mdl.get_count_by_topics()), key=lambda x:x[1], reverse=True)]
    for k in important_topics:
        if not mdl.is_live_topic(k): continue
        print('Topic #{}'.format(k))
        for word, prob in mdl.get_topic_words(k):
            print('\t', word, prob, sep='\t')

def word_prior_example(input_file):
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=corpus)
    # The word 'church' is assigned to Topic 0 with a weight of 1.0 and to the remaining topics with a weight of 0.1.
    # Therefore, a topic related to 'church' can be fixed at Topic 0 .
    mdl.set_word_prior('church', [1.0 if k == 0 else 0.1 for k in range(20)])
    # Topic 1 for a topic related to 'softwar'
    mdl.set_word_prior('softwar', [1.0 if k == 1 else 0.1 for k in range(20)])
    # Topic 2 for a topic related to 'citi'
    mdl.set_word_prior('citi', [1.0 if k == 2 else 0.1 for k in range(20)])
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()

def corpus_and_labeling_example(input_file):
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()

def raw_corpus_and_labeling_example(input_file):
    from nltk.stem.porter import PorterStemmer
    from nltk.corpus import stopwords
    stemmer = PorterStemmer()
    stops = set(stopwords.words('english'))
    corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(stemmer=stemmer.stem), 
        stopwords=lambda x: len(x) <= 2 or x in stops)
    # data_feeder yields a tuple of (raw string, user data) or a str (raw string)
    corpus.process(open(input_file, encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    
    # extract candidates for auto topic labeling
    extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
    cands = extractor.extract(mdl)

    labeler = tp.label.FoRelevance(mdl, cands, min_df=5, smoothing=1e-2, mu=0.25)
    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        print("Labels:", ', '.join(label for label, score in labeler.get_topic_labels(k, top_n=5)))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()


# You can get the sample data file 'enwiki-stemmed-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

print('Running LDA')
lda_example('enwiki-stemmed-1000.txt', 'test.lda.bin')

print('Running HDP')
hdp_example('enwiki-stemmed-1000.txt', 'test.hdp.bin')

print('Set Word Prior')
word_prior_example('enwiki-stemmed-1000.txt')

print('Running LDA and Labeling')
corpus_and_labeling_example('enwiki-stemmed-1000.txt')

print('Running LDA from raw corpus and Labeling')
raw_corpus_and_labeling_example('enwiki-1000.txt')
