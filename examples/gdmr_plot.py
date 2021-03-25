'''
This example show how to perform a g-DMR topic model using tomotopy
and visualize a topic distribution map.

Required Packages:
    matplotlib
'''

import tomotopy as tp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

class ExpNormalize(clr.Normalize):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        if vmin == vmax:
            result.fill(0)
        elif vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                    mask=mask)
            resdat = result.data
            resdat = 1 - np.exp(-2 * resdat / self.scale)
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

heat = clr.LinearSegmentedColormap.from_list('heat', 
    [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0), (1, 1, 0), (1, 0, 0), (1, 1, 1)],
    N=1024
)

'''
You can get the sample data file from https://github.com/bab2min/g-dmr/tree/master/data .
'''

corpus = tp.utils.Corpus()
for line in open('dataset2.txt', encoding='utf-8'):
    fd = line.strip().split()
    corpus.add_doc(fd[2:], numeric_metadata=list(map(float, fd[:2])))

# We set a range of the first metadata as [2000, 2017] 
# and one of the second metadata as [0, 1].
mdl = tp.GDMRModel(tw=tp.TermWeight.PMI, k=30, degrees=[4, 3], 
    alpha=1e-2, sigma=0.25, sigma0=3.0,
    metadata_range=[(2000, 2017), (0, 1)], corpus=corpus
)
mdl.optim_interval = 20
mdl.burn_in = 200

mdl.train(0)

print('Num docs:{}, Num Vocabs:{}, Total Words:{}'.format(
    len(mdl.docs), len(mdl.used_vocabs), mdl.num_words
))

# Let's train the model
for i in range(0, 1000, 20):
    print('Iteration: {:04} LL per word: {:.4}'.format(i, mdl.ll_per_word))
    mdl.train(20)
print('Iteration: {:04} LL per word: {:.4}'.format(1000, mdl.ll_per_word))

mdl.summary()

# Let's visualize the result
topic_counts = mdl.get_count_by_topics()
lambdas = mdl.lambdas

md_range = mdl.metadata_range
# Our topic distribution map has 
# 400 pixels for the first axis and 
# 200 pixels for the second axis.
r = mdl.tdf_linspace(
    [md_range[0][0], md_range[1][0]], 
    [md_range[0][1], md_range[1][1]], 
    [400, 200]
)

for k in (-topic_counts).argsort():
    print('Topic #{} ({})'.format(k, topic_counts[k]))
    print(*(w for w, _ in mdl.get_topic_words(k)))
    print('Lambda:', lambdas[k])

    imgplot = plt.imshow(r[:, :, k].transpose(), clim=(0.0, r[:, :, k].max()), 
        origin='lower', cmap=heat, norm=ExpNormalize(scale=0.04),
        extent=[*md_range[0], *md_range[1]],
        aspect='auto'
    )
    plt.title('#{}\n({})'.format(k, ' '.join(w for w, _ in mdl.get_topic_words(k, top_n=5))))
    plt.colorbar()
    plt.show()
