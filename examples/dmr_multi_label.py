'''
This example show how to perform a DMR topic model with multi-metadata using tomotopy
'''
import itertools

import tomotopy as tp
import numpy as np

# You can get the sample data file from https://github.com/bab2min/g-dmr/tree/master/data .
corpus = tp.utils.Corpus()
for line in open('text_mining_year_journal.txt', encoding='utf-8'):
    fd = line.strip().split('\t', maxsplit=2)
    corpus.add_doc(fd[2].split(), multi_metadata=['y_' + fd[0], 'j_' + fd[1]])
# We add prefix 'y' for year-label and 'j' for journal-label

# We set a range of the first metadata as [2000, 2017] 
# and one of the second metadata as [0, 1].
mdl = tp.DMRModel(tw=tp.TermWeight.ONE, 
    k=20,
    corpus=corpus
)
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
for i in range(0, 2000, 20):
    print('Iteration: {:04} LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04} LL per word: {:.4}'.format(2000, mdl.ll_per_word))

mdl.summary()

year_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('y_'))
journal_labels = sorted(l for l in mdl.multi_metadata_dict if l.startswith('j_'))

# calculate topic distribution with each metadata using get_topic_prior()
print('Topic distributions by year')
for l in year_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

print('Topic distributions by journal')
for l in journal_labels:
    print(l, '\n', mdl.get_topic_prior(multi_metadata=[l]), '\n')

# Also we can estimate topic distributions with multiple metadata
print('Topic distributions by year-journal')
for y, j in itertools.product(year_labels, journal_labels):
    print(y, ',', j, '\n', mdl.get_topic_prior(multi_metadata=[y, j]), '\n')
