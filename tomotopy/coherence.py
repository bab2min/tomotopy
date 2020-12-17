'''
..versionadded:: 0.10.0

This module provides the way to calculate topic coherence introduced the following paper:

> * Röder, M., Both, A., & Hinneburg, A. (2015, February). Exploring the space of topic coherence measures. In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399-408).
> http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
> https://github.com/dice-group/Palmetto

'''

from _tomotopy import _Coherence

from enum import IntEnum

class ProbEstimation(IntEnum):
    '''
    This enumeration follows the probability estimator in the paper.
    '''
    DOCUMENT = 1
    SLIDING_WINDOWS = 2

class Segmentation(IntEnum):
    '''
    This enumeration follows the segmentation in the paper.
    '''

    ONE_ONE = 1
    ONE_PRE = 2
    ONE_SUC = 3
    ONE_ALL = 4
    ONE_SET = 5

class ConfirmMeasure(IntEnum):
    '''
    This enumeration follows the direct confirm measure in the paper.
    '''

    DIFFERENCE = 1
    RATIO = 2
    LIKELIHOOD = 3
    LOGLIKELIHOOD = 4
    PMI = 5
    NPMI = 6
    LOGCOND = 7

class IndirectMeasure(IntEnum):
    '''
    This enumeration follows the indirect confirm measure in the paper.
    '''

    NONE = 0
    COSINE = 1
    DICE = 2
    JACCARD = 3

del IntEnum    

class Coherence(_Coherence):
    '''`Coherence` class provides the way to calculate topic coherence.
    '''
    _COH_MAP = {
        'u_mass':(ProbEstimation.DOCUMENT, 0, Segmentation.ONE_PRE, ConfirmMeasure.LOGCOND, IndirectMeasure.NONE),
        'c_v':(ProbEstimation.SLIDING_WINDOWS, 110, Segmentation.ONE_SET, ConfirmMeasure.NPMI, IndirectMeasure.COSINE),
        'c_uci':(ProbEstimation.SLIDING_WINDOWS, 10, Segmentation.ONE_ONE, ConfirmMeasure.PMI, IndirectMeasure.NONE),
        'c_npmi':(ProbEstimation.SLIDING_WINDOWS, 10, Segmentation.ONE_ONE, ConfirmMeasure.NPMI, IndirectMeasure.NONE)
    }

    def __init__(self, corpus, coherence='u_mass', window_size=0, targets=None, top_n=10, eps=1e-12, gamma=1.0):
        '''Initialize an instance to calculate coherence for given corpus

Parameters
----------
corpus : Union[tomotopy.utils.Corpus, tomotopy.LDAModel]
    A reference corpus to be used for estimating probability. 
    Supports not only an instance of `tomotopy.utils.Corpus`, but also any topic model instances including `tomotopy.LDAModel` and its descendants.
    If `corpus` is an instance of `tomotpy.utils.Corpus`, `targets` must be given too.
coherence : Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
    A coherence metric to be used. The metric can be a combination of (`tomotopy.coherence.ProbEstimation`, `tomotopy.coherence.Segmentation`, `tomotopy.coherence.ConfirmMeasure`)
    or a combination of (`tomotopy.coherence.ProbEstimation`, `tomotopy.coherence.Segmentation`, `tomotopy.coherence.ConfirmMeasure`, `tomotopy.coherence.IndirectMeasure`).
    
    Also shorthands in `str` type are supported as follows:
    > * 'u_mass' : (`tomotopy.coherence.ProbEstimation.DOCUMENT`, `tomotopy.coherence.Segmentation.ONE_PRE`, `tomotopy.coherence.ConfirmMeasure.LOGCOND`)
    > * 'c_uci' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_ONE`, `tomotopy.coherence.ConfirmMeasure.PMI`)
    > * 'c_npmi' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_ONE`, `tomotopy.coherence.ConfirmMeasure.NPMI`)
    > * 'c_v' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_SET`, `tomotopy.coherence.ConfirmMeasure.NPMI`, `tomotopy.coherence.IndirectMeasure.COSINE`)
window_size : int
    A window size for `tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`.
    default value is 10 for 'c_uci' and 'c_npmi' and 110 for 'c_v'.
targets : Iterable[str]
    If `corpus` is an instance of `tomotpy.utils.Corpus`, target words must be given. 
    Only words that are provided as `targets` are included in probability estimation.
top_n : int
    The number of top words to be extracted from each topic.
    If `corpus` is an instance of `tomotopy.LDAModel` and its descendants, target words are extracted from each topic of the topic model.
    But if `targets` are given, target words are extracted from `targets`, even if `corpus` is an instance of topic models.
eps : float
    An epsilon value to prevent division by zero
gamma : float
    A gamma value for indirect confirm measures
        '''
        import tomotopy as tp
        import itertools
        self._top_n = top_n
        if isinstance(corpus, tp.LDAModel):
            self._topic_model = corpus
            if targets is None:
                targets = itertools.chain(*((w for w, _ in corpus.get_topic_words(k, top_n=top_n)) for k in range(corpus.k)))
            corpus = corpus.docs
        else:
            self._topic_model = None
        
        try:
            pe, w, seg, cm, im = Coherence._COH_MAP[coherence]
        except KeyError:
            if type(coherence) is str: raise ValueError("Unknown `coherence` value (given {})".format(repr(coherence)))
            if len(coherence) not in (3, 4): raise ValueError("`coherence` must be a tuple with len=3 or len=4 (given {})".format(repr(coherence)))
            if len(coherence) == 3: 
                pe, seg, cm = coherence
                im = IndirectMeasure.NONE
            else:
                pe, seg, cm, im = coherence
            w = 0

            if type(pe) is str: pe = ProbEstimation[pe]
            if type(seg) is str: seg = Segmentation[seg]
            if type(cm) is str: cm = ConfirmMeasure[cm]
            if type(im) is str: im = IndirectMeasure[im]
        
        if not targets: raise ValueError("`targets` must be given as a non-empty iterable of str.")

        super().__init__(corpus, pe=pe, seg=seg, cm=cm, im=im, window_size=window_size or w, targets=targets, eps=eps, gamma=gamma)
    
    def get_score(self, words=None, topic_id=None):
        '''Calculate the coherence score for given `words` or `topic_id`

Parameters
----------
words : Iterable[str]
    Words whose coherence is calculated. 
    If `tomotopy.coherence.Coherence` was initialized using `corpus` as `tomotopy.LDAModel` or its descendants, `words` can be omitted.
    In this case words are extracted from topic with `topic_id`.
topic_id : int
    An id of the topic from which words are extracted. 
    This parameter is valid when `tomotopy.coherence.Coherence` was initialized using `corpus` as `tomotopy.LDAModel` or its descendants.
    If this is omitted, the average score of all topics is returned.
        '''
        if words is None and self._topic_model is None:
            raise ValueError("`words` must be provided if `Coherence` is not bound to an instance of topic model.")
        if words is None and topic_id is None:
            c = []
            for k in range(self._topic_model.k):
                c.append(super().get_score((w for w, _ in self._topic_model.get_topic_words(k, top_n=self._top_n))))
            return sum(c) / len(c)
        
        if words is None:
            words = (w for w, _ in self._topic_model.get_topic_words(topic_id, top_n=self._top_n))
        return super().get_score(words)
