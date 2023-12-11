'''
This example show how to perform a DMR topic model using tomotopy
and visualize the topic distribution for each metadata

Required Packages:
    matplotlib
'''

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt

'''
You can get the sample data file from https://drive.google.com/file/d/1AUHdwaPzw5qW0j8MaKqFNfw-SQDMbIzw/view?usp=sharing .
'''

corpus = tp.utils.Corpus()
for line in open('text_mining.txt', encoding='utf-8'):
    fd = line.strip().split('\t')
    corpus.add_doc(fd[1].lower().split(), metadata=fd[0])

# We set a range of the first metadata as [2000, 2017] 
# and one of the second metadata as [0, 1].
mdl = tp.DMRModel(tw=tp.TermWeight.PMI, 
    k=15,
    corpus=corpus
)
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
mdl.train(2000, show_progress=True)

mdl.summary()

# calculate topic distribution for each metadata using softmax
probs = np.exp(mdl.lambdas - mdl.lambdas.max(axis=0))
probs /= probs.sum(axis=0)

print('Topic distributions for each metadata')
for f, metadata_name in enumerate(mdl.metadata_dict):
    print(metadata_name, probs[:, f], '\n')

x = np.arange(mdl.k)
width = 1 / (mdl.f + 2)

fig, ax = plt.subplots()
for f, metadata_name in enumerate(mdl.metadata_dict):
    ax.bar(x + width * (f - mdl.f / 2), probs[:, f], width, label=mdl.metadata_dict[f])

ax.set_ylabel('Probabilities')
ax.set_yscale('log')
ax.set_title('Topic distributions')
ax.set_xticks(x)
ax.set_xticklabels(['Topic #{}'.format(k) for k in range(mdl.k)])
ax.legend()

fig.tight_layout()
plt.show()
