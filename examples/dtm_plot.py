import tomotopy as tp
import numpy as np
import nltk

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

num_timepoints = 13

mdl = tp.DTModel(min_cf=3, k=10, t=num_timepoints, phi_var=1e-2, corpus=corpus)
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

topic_dist_by_time = np.zeros(shape=[num_timepoints, mdl.k], dtype=np.float)
doc_counts_by_time = np.zeros(shape=[num_timepoints], dtype=np.int32)
for doc in mdl.docs:
    doc_counts_by_time[doc.timepoint] += 1
    topic_dist_by_time[doc.timepoint] += doc.get_topic_dist()

topic_dist_by_time /= doc_counts_by_time[:, np.newaxis]

for k in range(mdl.k):
    print('Topic #{}'.format(k), *(w for w, _ in mdl.get_topic_words(k, 0, top_n=5)))
    print(topic_dist_by_time[:, k])
    