import sys
import tomotopy as tp

# You can get the sample data file 'enwiki-stemmed-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

def infer_new_corpus():
    '''
    Since 0.10.0 version, inference using an instance of `Corpus` was supported.
    '''

    train_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    train_corpus.process(open('enwiki-stemmed-1000.txt', encoding='utf-8'))

    test_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    test_corpus.process(open('corpus_to_be_inferred.txt', encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=train_corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    
    mdl.train(1000, show_progress=True)
    mdl.summary()

    inferred_corpus, ll = mdl.infer(test_corpus)

    # print topic distributions of each document
    for doc in inferred_corpus:
        #print(doc.raw) # print raw string of the document
        #print(list(doc)) # print a list of words within the document
        print(doc.get_topic_dist())

def infer_new_doc():
    '''
    Prior to version 0.10.0, we had to make instances of `Document` using `make_doc` first
    and call `infer`.
    '''
    train_corpus = tp.utils.Corpus(tokenizer=tp.utils.SimpleTokenizer(), stopwords=['.'])
    train_corpus.process(open('enwiki-stemmed-1000.txt', encoding='utf-8'))

    # make LDA model and train
    mdl = tp.LDAModel(k=20, min_cf=10, min_df=5, corpus=train_corpus)
    mdl.train(0)
    print('Num docs:', len(mdl.docs), ', Vocab size:', len(mdl.used_vocabs), ', Num words:', mdl.num_words)
    print('Removed top words:', mdl.removed_top_words)
    for i in range(0, 1000, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

    mdl.summary()

    docs = []
    for line in open('enwiki-stemmed-1000.txt', encoding='utf-8'):
        docs.append(mdl.make_doc(line.lower().split()))

    topic_distributions, ll = mdl.infer(docs)

    # print topic distributions of each document
    for doc, topic_dist in zip(docs, topic_distributions):
        #print(doc)
        print(topic_dist)

infer_new_corpus()
infer_new_doc()