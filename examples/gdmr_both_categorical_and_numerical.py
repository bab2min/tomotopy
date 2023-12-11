'''
This example show how to perform a g-DMR topic model
for mixture of categorical and numerical metadata using tomotopy
and visualize a topic distribution.

Required Packages:
    matplotlib
'''

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import re

#You can get the sample data file from https://github.com/bab2min/g-dmr/tree/master/data .
corpus = tp.utils.Corpus()
for line in open('text_mining_year_journal.txt', encoding='utf-8'):
    fd = line.strip().split('\t', maxsplit=2)
    corpus.add_doc(fd[2].split(), numeric_metadata=[float(fd[0])], metadata=fd[1])
# Use the argument `numeric_metadata` for continuous numerical metadata (list of float type),
# and the argument `metadata` for categorical metadata (str type)

# We set a range of the numeric metadata as [2000, 2017].
# `decay=1.0` penalizes higher-order terms of lambdas to prevent overfitting.
mdl = tp.GDMRModel(tw=tp.TermWeight.ONE, k=30, degrees=[6], 
    alpha=1e-2, sigma=0.25, sigma0=3.0, decay=1.0,
    metadata_range=[(2000, 2017)], corpus=corpus
)
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
mdl.train(1000, show_progress=True)
mdl.summary()

# Let's visualize the result
topic_counts = mdl.get_count_by_topics()
lambdas = mdl.lambdas
lambdas = lambdas.reshape(lambdas.shape[:1] + (len(mdl.metadata_dict), -1))
# lambdas shape: [num_topics, num_categorical_metadata, degrees + 1]

md_range = mdl.metadata_range
r = np.stack([mdl.tdf_linspace(
    [md_range[0][0]], 
    [md_range[0][1]], 
    [50], # interpolation size
    cat
) for cat in mdl.metadata_dict])
# r shape: [num_categorical_metadata, 50, num_topics]

xs = np.linspace(*md_range[0], 50)
for k in (-topic_counts).argsort():
    print('Topic #{} ({})'.format(k, topic_counts[k]))
    print(*(w for w, _ in mdl.get_topic_words(k)))
    print('Lambda:', lambdas[k].reshape((len(mdl.metadata_dict), -1)))

    for label, ys in zip(mdl.metadata_dict, r[:, :, k]):
        label = re.sub(r'^(Proceedings|Journal)( of)?( the)?( -)?|International Conference on', '', label).strip()
        if len(label) >= 35: label = label[:33] + '...'
        plt.plot(xs, ys, linewidth=2, label=label)
    plt.title('#{}\n({})'.format(k, ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=5))))
    plt.legend()
    plt.show()
