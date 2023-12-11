import sys
import tomotopy as tp

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
    mdl.train(1000, show_progress=True)
    mdl.summary()

    for k in range(mdl.k):
        print("== Topic #{} ==".format(k))
        for word, prob in mdl.get_topic_words(k, top_n=10):
            print(word, prob, sep='\t')
        print()


# You can get the sample data file 'enwiki-stemmed-1000.txt'
# at https://drive.google.com/file/d/18OpNijd4iwPyYZ2O7pQoPyeTAKEXa71J/view?usp=sharing

print('Set Word Prior')
word_prior_example('enwiki-stemmed-1000.txt')
