"""
Submodule `tomotopy.label` provides automatic topic labeling techniques.
You can label topics automatically with simple code like below. The results are attached to the bottom of the code.

.. include:: ./auto_labeling_code.rst
"""

from typing import List, Tuple

from _tomotopy import (_LabelCandidate, _LabelPMIExtractor, _LabelFoRelevance)

Candidate = _LabelCandidate

class PMIExtractor(_LabelPMIExtractor):
    def __init__(self, min_cf=10, min_df=5, min_len=1, max_len=5, max_cand=5000, normalized=False):
        '''.. versionadded:: 0.6.0

`PMIExtractor` exploits multivariate pointwise mutual information to extract collocations. 
It finds a string of words that often co-occur statistically.

Parameters
----------
min_cf : int
    minimum collection frequency of collocations. Collocations with a smaller collection frequency than `min_cf` are excluded from the candidates.
    Set this value large if the corpus is big
min_df : int
    minimum document frequency of collocations. Collocations with a smaller document frequency than `min_df` are excluded from the candidates.
    Set this value large if the corpus is big
min_len : int
    .. versionadded:: 0.10.0
    
    minimum length of collocations. `min_len=1` means that it extracts not only collocations but also all single words.
    The number of single words are excluded in counting `max_cand`.
max_len : int
    maximum length of collocations
max_cand : int
    maximum number of candidates to extract
'''
        super().__init__(min_cf, min_df, min_len, max_len, max_cand, normalized)

    def extract(self, topic_model) -> List:
        '''Return the list of `tomotopy.label.Candidate`s extracted from `topic_model`

Parameters
----------
topic_model
    an instance of topic model with documents to extract candidates
'''
        return super().extract(topic_model)

class FoRelevance(_LabelFoRelevance):
    def __init__(self, topic_model, cands, min_df=5, smoothing=0.01, mu=0.25, window_size=-1, workers=0):
        '''.. versionadded:: 0.6.0

This type provides an implementation of First-order Relevance for topic labeling based on following papers:

> * Mei, Q., Shen, X., & Zhai, C. (2007, August). Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499).

Parameters
----------
topic_model
    an instance of topic model to label topics
cands : Iterable[tomotopy.label.Candidate]
    a list of candidates to be used as topic labels
min_df : int
    minimum document frequency of collocations. Collocations with a smaller document frequency than `min_df` are excluded from the candidates.
    Set this value large if the corpus is big
smoothing : float
    a small value greater than 0 for Laplace smoothing
mu : float
    a discriminative coefficient. Candidates with high score on a specific topic and with low score on other topics get the higher final score when this value is the larger.
window_size : int
    .. versionadded:: 0.10.0
    
    size of the sliding window for calculating co-occurrence. If `window_size=-1`, it uses the whole document, instead of the sliding windows.
    If your documents are long, it is recommended to set this value to 50 ~ 100, not -1.
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
'''
        super().__init__(topic_model, cands, min_df, smoothing, mu, window_size, workers)
    
    def get_topic_labels(self, k, top_n=10) -> List[Tuple[str, float]]:
        '''Return the top-n label candidates for the topic `k`

Parameters
----------
k : int
    an integer indicating a topic
top_n : int
    the number of labels
'''
        return super().get_topic_labels(k, top_n)

