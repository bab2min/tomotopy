import tomotopy as tp
import numpy as np
import nltk
import pyLDAvis

def data_feeder(input_file):
    for line in open(input_file, encoding='utf-8'):
        fd = line.strip().split(maxsplit=1)
        timepoint = int(fd[0])
        yield fd[1], None, {'timepoint':timepoint}

porter_stemmer = nltk.PorterStemmer().stem
corpus = tp.utils.Corpus(
    tokenizer=tp.utils.SimpleTokenizer(porter_stemmer)
)
corpus.process(data_feeder('../test/sample_tp.txt'))

mdl = tp.DTModel(min_cf=3, k=10, t=13, phi_var=1e-2, corpus=corpus)
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

topic_dist_by_time = np.zeros(shape=[mdl.num_timepoints, mdl.k], dtype=np.float)
for doc in mdl.docs:
    topic_dist_by_time[doc.timepoint] += doc.get_topic_dist()

topic_dist_by_time /= mdl.num_docs_by_timepoint[:, np.newaxis]

for k in range(mdl.k):
    print('Topic #{}'.format(k), *(w for w, _ in mdl.get_topic_words(k, 0, top_n=5)))
    print(topic_dist_by_time[:, k])

for timepoint in range(mdl.num_timepoints):
    topic_term_dists = np.stack([mdl.get_topic_word_dist(k, timepoint=timepoint) for k in range(mdl.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in mdl.docs if doc.timepoint == timepoint])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    doc_lengths = np.array([len(doc.words) for doc in mdl.docs if doc.timepoint == timepoint])
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
    pyLDAvis.save_html(prepared_data, 'dtmvis_{}.html'.format(timepoint))
