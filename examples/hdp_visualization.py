'''
This example shows how to perform a Latent Dirichlet Allocation using tomotopy 
and visualize the result.


Required Packages:
    nltk, sklearn, pyldavis
'''

import tomotopy as tp
import nltk
from nltk.corpus import stopwords
import re
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pyLDAvis

try:
    # load if preprocessed corpus exists
    corpus = tp.utils.Corpus.load('preprocessed_20news.cps')
except IOError:
    porter_stemmer = nltk.PorterStemmer().stem
    english_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
    pat = re.compile('^[a-z]{2,}$')
    corpus = tp.utils.Corpus(
        tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
        stopwords=lambda x: x in english_stops or not pat.match(x)
    )
    newsgroups_train = fetch_20newsgroups()
    corpus.process(d.lower() for d in newsgroups_train.data)
    # save preprocessed corpus for reuse
    corpus.save('preprocessed_20news.cps')

mdl = tp.HDPModel(tw=tp.TermWeight.PMI, min_df=5, rm_top=30, alpha=1, gamma=10, initial_k=10, corpus=corpus)
mdl.train(0)
mdl.burn_in = 500

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))
print('Removed Top words: ', *mdl.removed_top_words)

# Let's train the model
for i in range(0, 5000, 50):
    print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(50)
print('Iteration: {:04}, LL per word: {:.4}'.format(mdl.global_step, mdl.ll_per_word))

mdl.summary()

live_topics = [k for k in range(mdl.k) if mdl.is_live_topic(k)]

topic_term_dists = np.stack([mdl.get_topic_word_dist(k) for k in range(mdl.k)])
topic_term_dists = topic_term_dists[live_topics]
topic_term_dists /= topic_term_dists.sum(axis=1, keepdims=True)

doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs])
doc_topic_dists = doc_topic_dists[:, live_topics]
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)

doc_lengths = np.array([len(doc.words) for doc in mdl.docs])
vocab = list(mdl.used_vocabs)
term_frequency = mdl.used_vocab_freq

prepared_data = pyLDAvis.prepare(
    topic_term_dists, 
    doc_topic_dists, 
    doc_lengths, 
    vocab, 
    term_frequency,
    start_index=0,
    sort_topics=False
)
pyLDAvis.save_html(prepared_data, 'ldavis.html')
