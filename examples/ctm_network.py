'''
This example shows how to perform a Correlated Topic Model using tomotopy 
and visualize the correlation between topics.


Required Packages:
    nltk, sklearn, pyvis
'''

import tomotopy as tp
import nltk
from nltk.corpus import stopwords
import re
from sklearn.datasets import fetch_20newsgroups
from pyvis.network import Network

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

mdl = tp.CTModel(tw=tp.TermWeight.IDF, min_df=5, rm_top=40, k=30, corpus=corpus)
mdl.train(0)

# Since we have more than ten thousand of documents, 
# setting the `num_beta_sample` smaller value will not cause an inaccurate result.
mdl.num_beta_sample = 5
print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))
print('Removed Top words: ', *mdl.removed_top_words)

# Let's train the model
for i in range(0, 1000, 20):
    print('Iteration: {:04}, LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04}, LL per word: {:.4}'.format(1000, mdl.ll_per_word))

# Let's visualize the result
g = Network(width=800, height=800, font_color="#333")
correl = mdl.get_correlations().reshape([-1])
correl.sort()
top_tenth = mdl.k * (mdl.k - 1) // 10
top_tenth = correl[-mdl.k - top_tenth]

for k in range(mdl.k):
    label = "#{}".format(k)
    title= ' '.join(word for word, _ in mdl.get_topic_words(k, top_n=6))
    print('Topic', label, title)
    g.add_node(k, label=label, title=title, shape='ellipse')
    for l, correlation in zip(range(k - 1), mdl.get_correlations(k)):
        if correlation < top_tenth: continue
        g.add_edge(k, l, value=float(correlation), title='{:.02}'.format(correlation))

g.barnes_hut(gravity=-1000, spring_length=20)
g.show_buttons()
g.show("topic_network.html")
