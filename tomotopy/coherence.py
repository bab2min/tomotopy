
def _load():
    import importlib, os
    env_setting = os.environ.get('TOMOTOPY_ISA', '').split(',')
    if not env_setting[0]: env_setting = []
    if len(env_setting) == 0 or len(env_setting) > 1:
        from cpuinfo import get_cpu_info
        flags = get_cpu_info()['flags']
    else:
        flags = []
    isas = ['avx2', 'avx', 'sse2', 'none']
    isas = [isa for isa in isas if (env_setting and isa in env_setting) or (not env_setting and (isa in flags or isa == 'none'))]
    if not isas: raise RuntimeError("No isa option for " + str(env_setting))
    for isa in isas:
        try:
            mod_name = '_tomotopy' + ('_' + isa if isa != 'none' else '')
            globals().update({k:v for k, v in vars(importlib.import_module(mod_name)).items() if k in ('_Coherence', )})
            return
        except:
            if isa == isas[-1]: raise
_load()

from enum import IntEnum

class ProbEstimation(IntEnum):
    DOCUMENT = 1
    SLIDING_WINDOWS = 2

class Segmentation(IntEnum):
    ONE_ONE = 1
    ONE_PRE = 2
    ONE_SUC = 3
    ONE_ALL = 4
    ONE_SET = 5

class ConfirmMeasure(IntEnum):
    DIFFERENCE = 1
    RATIO = 2
    LIKELIHOOD = 3
    LOGLIKELIHOOD = 4
    PMI = 5
    NPMI = 6
    LOGCOND = 7


class IndirectMeasure(IntEnum):
    NONE = 0
    COSINE = 1
    DICE = 2
    JACCARD = 3
    
class Coherence(_Coherence):
    _COH_MAP = {
        'u_mass':(ProbEstimation.DOCUMENT, 0, Segmentation.ONE_PRE, ConfirmMeasure.LOGCOND, IndirectMeasure.NONE),
        'c_v':(ProbEstimation.SLIDING_WINDOWS, 110, Segmentation.ONE_SET, ConfirmMeasure.NPMI, IndirectMeasure.COSINE),
        'c_uci':(ProbEstimation.SLIDING_WINDOWS, 10, Segmentation.ONE_ONE, ConfirmMeasure.PMI, IndirectMeasure.NONE),
        'c_npmi':(ProbEstimation.SLIDING_WINDOWS, 10, Segmentation.ONE_ONE, ConfirmMeasure.NPMI, IndirectMeasure.NONE)
    }

    def __init__(self, corpus, coherence=None, window_size=0, targets=None, top_n=10, eps=1e-12, gamma=1.0):
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

        super().__init__(corpus, pe=pe, seg=seg, cm=cm, im=im, window_size=window_size or w, targets=targets, eps=eps, gamma=gamma)
    
    def get_score(self, words=None, topic_id=None):
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
