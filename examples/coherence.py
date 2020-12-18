'''
This example shows how to perform a Latent Dirichlet Allocation 
and calculate coherence of the results.

Required Packages:
    nltk, sklearn
'''

import tomotopy as tp
import nltk
from nltk.corpus import stopwords
import re
from sklearn.datasets import fetch_20newsgroups
import itertools

print('Training lda models...')
try:
    # load if trained model exist already
    mdl = tp.LDAModel.load('trained_lda_model.bin')
except:
    porter_stemmer = nltk.PorterStemmer().stem
    english_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
    pat = re.compile('^[a-z]{2,}$')
    corpus = tp.utils.Corpus(
        tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
        stopwords=lambda x: x in english_stops or not pat.match(x)
    )
    newsgroups_train = fetch_20newsgroups()
    corpus.process(d.lower() for d in newsgroups_train.data)

    mdl = tp.LDAModel(min_df=5, rm_top=30, k=20, corpus=corpus)
    mdl.train(0)

    print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
        len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
    ))
    print('Removed Top words: ', *mdl.removed_top_words)

    # Let's train the model
    for i in range(0, 1000, 20):
        print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
        mdl.train(20)
    print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))

    mdl.summary()

    # save lda model for reuse
    mdl.save('trained_lda_model.bin')

# calculate coherence using preset
for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
    coh = tp.coherence.Coherence(mdl, coherence=preset)
    average_coherence = coh.get_score()
    coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
    print('==== Coherence : {} ===='.format(preset))
    print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
    print()

# calculate coherence using custom combination
for seg, cm, im in itertools.product(tp.coherence.Segmentation, tp.coherence.ConfirmMeasure, tp.coherence.IndirectMeasure):
    coh = tp.coherence.Coherence(mdl, coherence=(tp.coherence.ProbEstimation.DOCUMENT, seg, cm, im))
    average_coherence = coh.get_score()
    coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
    print('==== Coherence : {}, {}, {} ===='.format(repr(seg), repr(cm), repr(im)))
    print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
    print()
