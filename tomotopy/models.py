'''
Submodule `tomotopy.models` provides various topic model classes.
All models are based on `tomotopy.models.LDAModel`, which implements the basic Latent Dirichlet Allocation.
Derived models include DMR, GDMR, HDP, MGLDA, PA, HPA, CT, SLDA, LLDA, PLDA, HLDA, DT and PT.
'''

from typing import Optional, Union, List, Tuple
from copy import deepcopy
import re
import pickle
from collections import Counter

import numpy as np

from _tomotopy import (
    _LDAModel,
    _DMRModel,
    _GDMRModel,
    _HDPModel,
    _MGLDAModel,
    _PAModel,
    _HPAModel,
    _CTModel,
    _SLDAModel,
    _LLDAModel,
    _PLDAModel,
    _HLDAModel,
    _DTModel,
    _PTModel,
)

from tomotopy._version import __version__
from tomotopy.utils import Corpus, Document

def _convert_term_weight(tw):
    from tomotopy import TermWeight
    if isinstance(tw, str):
        try:
            return getattr(TermWeight, tw.upper())
        except AttributeError:
            raise ValueError(f'Unknown term weight: {tw}')
    return tw

def _format_numpy(arr, prefix=''):
    arr = np.array(arr)
    return ('\n' + prefix).join(str(arr).split('\n'))

class LDAModel(_LDAModel):
    '''This type provides Latent Dirichlet Allocation(LDA) topic model and its implementation is based on the following papers:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.'''

    def __init__(self, 
                 tw: Union[int, str] ='one',
                 min_cf: int = 0,
                 min_df: int = 0,
                 rm_top: int = 0,
                 k: int = 1,
                 alpha: Union[float, List[float]] = 0.1,
                 eta: float = 0.01,
                 seed: Optional[int] = None,
                 corpus = None,
                 transform = None,
                 ):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            alpha,
            eta,
            seed,
            corpus,
            transform,
        )
    
    @classmethod
    def load(cls, filename: str) -> 'LDAModel':
        '''Return the model instance loaded from file `filename`.'''
        inst, extra_data = cls._load(cls, filename)
        inst.init_params = pickle.loads(extra_data)
        return inst
    
    @classmethod
    def loads(cls, data: bytes) -> 'LDAModel':
        '''Return the model instance loaded from `data` in a bytes-like object.'''
        inst, extra_data = cls._loads(cls, data)
        inst.init_params = pickle.loads(extra_data)
        return inst
    
    @property
    def alpha(self) -> Union[float, List[float]]:
        '''Dirichlet prior on the per-document topic distributions (read-only)'''
        return self._alpha
    
    @property
    def burn_in(self) -> int:
        '''get or set the burn-in iterations for optimizing parameters

Its default value is 0.'''
        return self._burn_in
    
    @burn_in.setter
    def burn_in(self, value: int):
        self._burn_in = value
    
    @property
    def docs(self):
        '''a `list`-like interface of `tomotopy.utils.Document` in the model instance (read-only)'''
        return self._docs
    
    @property
    def eta(self) -> float:
        '''the hyperparameter eta (read-only)'''
        return self._eta
    
    @property
    def global_step(self) -> int:
        '''the total number of iterations of training (read-only)

.. versionadded:: 0.9.0'''
        return self._global_step
    
    @property
    def k(self) -> int:
        '''K, the number of topics (read-only)'''
        return self._k
    
    @property
    def ll_per_word(self) -> float:
        '''a log likelihood per-word of the model (read-only)'''
        return self._ll_per_word
    
    @property
    def num_vocabs(self) -> int:
        '''the number of vocabularies after words with a smaller frequency were removed (read-only)

This value is 0 before `train` is called.

.. deprecated:: 0.8.0

    Due to the confusion of its name, this property will be removed. Please use `len(used_vocabs)` instead.'''
        return self._num_vocabs
    
    @property
    def num_words(self) -> int:
        '''the number of total words (read-only)

This value is 0 before `train` is called.'''
        return self._num_words
    
    @property
    def optim_interval(self) -> int:
        '''get or set the interval for optimizing parameters

Its default value is 10. If it is set to 0, the parameter optimization is turned off.'''
        return self._optim_interval
    
    @optim_interval.setter
    def optim_interval(self, value: int):
        self._optim_interval = value
    
    @property
    def perplexity(self) -> float:
        '''a perplexity of the model (read-only)'''
        return self._perplexity
    
    @property
    def removed_top_words(self) -> List[str]:
        '''a `list` of `str` which is a word removed from the model if you set `rm_top` greater than 0 at initializing the model (read-only)'''
        return self._removed_top_words
    
    @property
    def tw(self) -> int:
        '''the term weighting scheme (read-only)'''
        return self._tw
    
    @property
    def used_vocab_df(self) -> List[int]:
        '''a `list` of vocabulary document-frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0'''
        return self._used_vocab_df
    
    @property
    def used_vocab_freq(self) -> List[int]:
        '''a `list` of vocabulary frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0'''
        return self._used_vocab_freq
    
    @property
    def used_vocab_weighted_freq(self) -> List[float]:
        '''a `list` of term-weighted vocabulary frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.12.1'''
        return self._used_vocab_weighted_freq
    
    @property
    def used_vocabs(self):
        '''a dictionary, which contains only the vocabularies actually used in modeling, as the type `tomotopy.Dictionary` (read-only)

.. versionadded:: 0.8.0'''
        return self._used_vocabs
    
    @property
    def vocab_df(self) -> List[int]:
        '''a `list` of vocabulary document-frequencies which contains both vocabularies filtered by frequency and vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0'''
        return self._vocab_df
    
    @property
    def vocab_freq(self) -> List[int]:
        '''a `list` of vocabulary frequencies which contains both vocabularies filtered by frequency and vocabularies actually used in modeling (read-only)'''
        return self._vocab_freq
    
    @property
    def vocabs(self):
        '''a dictionary, which contains both vocabularies filtered by frequency and vocabularies actually used in modeling, as the type `tomotopy.Dictionary` (read-only)'''
        return self._vocabs
    
    def add_corpus(self, corpus, transform=None) -> Corpus:
        '''.. versionadded:: 0.10.0

Add new documents into the model instance using `tomotopy.utils.Corpus` and return an instance of corpus that contains the inserted documents. 
This method should be called before calling the `tomotopy.models.LDAModel.train`.

Parameters
----------
corpus : tomotopy.utils.Corpus
    corpus that contains documents to be added
transform : Callable[dict, dict]
    a callable object to manipulate arbitrary keyword arguments for a specific topic model
'''
        return self._add_corpus(corpus, transform)
    
    def add_doc(self, words, ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance and return an index of the inserted document. This method should be called before calling the `tomotopy.models.LDAModel.train`.

.. versionchanged:: 0.12.3

    A new argument `ignore_empty_words` was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, ignore_empty_words)
    
    def copy(self) -> 'LDAModel':
        '''.. versionadded:: 0.12.0

Return a new deep-copied instance of the current instance'''
        return self._copy(type(self))
    
    def get_count_by_topics(self) -> List[int]:
        '''Return the number of words allocated to each topic.'''
        return self._get_count_by_topics()
    
    def get_hash(self) -> int:
        return self._get_hash()
    
    def get_topic_word_dist(self, topic_id, normalize=True) -> List[float]:
        '''Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
'''
        return self._get_topic_word_dist(topic_id, normalize)
    
    def get_topic_words(self, topic_id, top_n=10, return_id=False) -> Union[List[Tuple[str, float]], List[Tuple[str, int, float]]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`) tuples if return_id is False,
otherwise a `list` of (word:`str`, word_id:`int`, probability:`float`) tuples.

Parameters
----------
topic_id : int
    an integer in range [0, `k`), indicating the topic
top_n : int
	the number of words to be returned
return_id : bool
	If `True`, it returns the word IDs too.
'''
        return self._get_topic_words(topic_id, top_n, return_id)
    
    def get_word_forms(self, idx = -1):
        return self._get_word_forms(idx)
    
    def get_word_prior(self, word) -> List[float]:
        '''.. versionadded:: 0.6.0

Return word-topic prior for `word`. If there is no set prior for `word`, an empty list is returned.

Parameters
----------
word : str
    a word
'''
        return self._get_word_prior(word)
    
    def infer(self, doc, iterations=100, tolerance=-1, workers=0, parallel=0, together=False, transform=None) -> Tuple[Union[List[float], List[List[float]], Corpus], List[float]]:
        '''Return the inferred topic distribution from unseen `doc`s.

Parameters
----------
doc : Union[tomotopy.utils.Document, Iterable[tomotopy.utils.Document], tomotopy.utils.Corpus]
    an instance of `tomotopy.utils.Document` or a `list` of instances of `tomotopy.utils.Document` to be inferred by the model.
    It can be acquired from `tomotopy.models.LDAModel.make_doc` method.

    .. versionchanged:: 0.10.0

        Since version 0.10.0, `infer` can receive a raw corpus instance of `tomotopy.utils.Corpus`. 
        In this case, you don't need to call `make_doc`. `infer` would generate documents bound to the model, estimate its topic distributions and
        return a corpus containing generated documents as the result.
iterations : int
    an integer indicating the number of iteration to estimate the distribution of topics of `doc`.
    The higher value will generate a more accurate result.
tolerance : float
    This parameter is not currently used.
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for inference. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.
together : bool
    all `doc`s are inferred together in one process if True, otherwise each `doc` is inferred independently. Its default value is `False`.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    a callable object to manipulate arbitrary keyword arguments for a specific topic model. 
    Available when `doc` is given as an instance of `tomotopy.utils.Corpus`.

Returns
-------
result : Union[List[float], List[List[float]], tomotopy.utils.Corpus]
    If `doc` is given as a single `tomotopy.utils.Document`, `result` is a single `List[float]` indicating its topic distribution.
    
    If `doc` is given as a list of `tomotopy.utils.Document`s, `result` is a list of `List[float]` indicating topic distributions for each document.
    
    If `doc` is given as an instance of `tomotopy.utils.Corpus`, `result` is another instance of `tomotopy.utils.Corpus` which contains inferred documents.
    You can get topic distribution for each document using `tomotopy.utils.Document.get_topic_dist`.
log_ll : List[float]
    a list of log-likelihoods for each `doc`
'''
        return self._infer(doc, iterations, tolerance, workers, parallel, together, transform)
    
    def make_doc(self, words) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
'''
        return self._make_doc(words)
    
    def save(self, filename: str, full=True) -> None:
        '''Save the model instance to file `filename`. Return `None`.

If `full` is `True`, the model with its all documents and state will be saved. If you want to train more after, use full model.
If `False`, only topic parameters of the model will be saved. This model can be only used for inference of an unseen document.

.. versionadded:: 0.6.0

Since version 0.6.0, the model file format has been changed. 
Thus model files saved in version 0.6.0 or later are not compatible with versions prior to 0.5.2.
'''
        extra_data = pickle.dumps(self.init_params)
        return self._save(filename, extra_data, full)
    
    def saves(self, full=True) -> bytes:
        '''.. versionadded:: 0.11.0

Serialize the model instance into `bytes` object and return it. The arguments work the same as `tomotopy.models.LDAModel.save`.'''
        extra_data = pickle.dumps(self.init_params)
        return self._saves(extra_data, full)
    
    def set_word_prior(self, word, prior) -> None:
        '''.. versionadded:: 0.6.0

Set word-topic prior. This method should be called before calling the `tomotopy.models.LDAModel.train`.

Parameters
----------
word : str
    a word to be set
prior : Union[Iterable[float], Dict[int, float]]
	topic distribution of `word` whose length is equal to `tomotopy.models.LDAModel.k`

Note
----
Since version 0.12.6, this method can accept a dictionary type parameter as well as a list type parameter for `prior`.
The key of the dictionary is the topic id and the value is the prior of the topic. If the prior of a topic is not set, the default value is set to `eta` parameter of the model.
```python
>>> model = tp.LDAModel(k=3, eta=0.01)
>>> model.set_word_prior('apple', [0.01, 0.9, 0.01])
>>> model.set_word_prior('apple', {1: 0.9}) # same effect as above
```
'''
        return self._set_word_prior(word, prior)
    
    @classmethod
    def _summary_extract_param_desc(cls:type):
        doc_string = cls.__doc__ or cls.__init__.__doc__
        if not doc_string: return {}
        ps = doc_string.split('\nParameters\n')[1].split('\n')
        param_name = re.compile(r'^([a-zA-Z0-9_]+)\s*:\s*')
        directive = re.compile(r'^\s*\.\.')
        descriptive = re.compile(r'\s+([^\s].*)')
        period = re.compile(r'[.,](\s|$)')
        ret = {}
        name = None
        desc = ''
        for p in ps:
            if directive.search(p): continue
            m = param_name.search(p)
            if m:
                if name: ret[name] = desc.split('. ')[0]
                name = m.group(1)
                desc = ''
                continue
            m = descriptive.search(p)
            if m:
                desc += (' ' if desc else '') + m.group(1)
                continue
        if name: ret[name] = period.split(desc)[0]
        return ret

    def _summary_basic_info(self, file):
        p = self.used_vocab_freq
        p = p / p.sum()
        entropy = -(p * np.log(p + 1e-20)).sum()

        p = self.used_vocab_weighted_freq
        p /= p.sum()
        w_entropy = -(p * np.log(p + 1e-20)).sum()

        print('| {} (current version: {})'.format(type(self).__name__, __version__), file=file)
        print('| {} docs, {} words'.format(len(self.docs), self.num_words), file=file)
        print('| Total Vocabs: {}, Used Vocabs: {}'.format(len(self.vocabs), len(self.used_vocabs)), file=file)
        print('| Entropy of words: {:.5f}'.format(entropy), file=file)
        print('| Entropy of term-weighted words: {:.5f}'.format(w_entropy), file=file)
        print('| Removed Vocabs: {}'.format(' '.join(self.removed_top_words) if self.removed_top_words else '<NA>'), file=file)

    def _summary_training_info(self, file):
        print('| Iterations: {}, Burn-in steps: {}'.format(self.global_step, self.burn_in), file=file)
        print('| Optimization Interval: {}'.format(self.optim_interval), file=file)
        print('| Log-likelihood per word: {:.5f}'.format(self.ll_per_word), file=file)

    def _summary_initial_params_info(self, file):
        param_desc = self._summary_extract_param_desc()
        if hasattr(self, 'init_params'):
            for k, v in self.init_params.items():
                if type(v) is float: fmt = ':.5'
                else: fmt = ''

                try:
                    getattr(self, f'_summary_initial_params_info_{k}')(v, file)
                except AttributeError:
                    if k in param_desc:
                        print(('| {}: {' + fmt + '} ({})').format(k, v, param_desc[k]), file=file)
                    else:
                        print(('| {}: {' + fmt + '}').format(k, v), file=file)
        else:
            print('| Not Available (The model seems to have been built in version < 0.9.0.)', file=file)

    def _summary_initial_params_info_tw(self, v, file):
        from tomotopy import TermWeight
        try:
            if isinstance(v, str):
                v = TermWeight[v.upper()].name
            else:
                v = TermWeight(v).name
        except:
            pass
        print('| tw: TermWeight.{}'.format(v), file=file)

    def _summary_initial_params_info_version(self, v, file):
        print('| trained in version {}'.format(v), file=file)

    def _summary_params_info(self, file):
        print('| alpha (Dirichlet prior on the per-document topic distributions)\n'
            '|  {}'.format(_format_numpy(self.alpha, '|  ')), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        for k in range(self.k):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('| #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

    def summary(self, initial_hp=True, params=True, topic_word_top_n=5, file=None, flush=False) -> None:
        '''.. versionadded:: 0.9.0

Print human-readable description of the current model

Parameters
----------
initial_hp : bool
    whether to show the initial parameters at model creation
params : bool
    whether to show the current parameters of the model
topic_word_top_n : int
    the number of words by topic to display
file
    a file-like object (stream), default is `sys.stdout`
flush : bool
    whether to forcibly flush the stream
'''
        flush = flush or False

        print('<Basic Info>', file=file)
        self._summary_basic_info(file=file)
        print('|', file=file)
        print('<Training Info>', file=file)
        self._summary_training_info(file=file)
        print('|', file=file)

        if initial_hp:
            print('<Initial Parameters>', file=file)
            self._summary_initial_params_info(file=file)
            print('|', file=file)
        
        if params:
            print('<Parameters>', file=file)
            self._summary_params_info(file=file)
            print('|', file=file)

        if topic_word_top_n > 0:
            print('<Topics>', file=file)
            self._summary_topics_info(file=file, topic_word_top_n=topic_word_top_n)
            print('|', file=file)

        print(file=file, flush=flush)

    
    def train(self, iterations=10, workers=0, parallel=0, freeze_topics=False, callback_interval=10, callback=None, show_progress=False) -> None:
        '''Train the model using Gibbs-sampling with `iterations` iterations. Return `None`. 
After calling this method, you cannot `tomotopy.models.LDAModel.add_doc` or `tomotopy.models.LDAModel.set_word_prior` more.

Parameters
----------
iterations : int
    the number of iterations of Gibbs-sampling
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for training. the default value is `tomotopy.ParallelScheme.DEFAULT` which means that tomotopy selects the best scheme by model.
freeze_topics : bool
    .. versionadded:: 0.10.1

    prevents creating a new topic when training. Only valid for `tomotopy.models.HLDAModel`
callback_interval : int
    .. versionadded:: 0.12.6

    the interval of calling `callback` function. If `callback_interval` <= 0, `callback` function is called at the beginning and the end of training.
callback : Callable[[tomotopy.models.LDAModel, int, int], None]
    .. versionadded:: 0.12.6

    a callable object which is called every `callback_interval` iterations. 
    It receives three arguments: the current model, the current number of iterations, and the total number of iterations.
show_progress : bool
    .. versionadded:: 0.12.6

    If `True`, it shows progress bar during training using `tqdm` package.
'''
        if show_progress:
            if callback is not None:
                callback = LDAModel._show_progress
            else:
                def _multiple_callbacks(*args):
                    callback(*args)
                    LDAModel._show_progress(*args)
                callback = _multiple_callbacks
        return self._train(iterations, workers, parallel, freeze_topics, callback_interval, callback)
    
    def _init_tqdm(self, current_iteration:int, total_iteration:int):
        from tqdm import tqdm
        self._tqdm = tqdm(total=total_iteration, desc='Iteration')
    
    def _close_tqdm(self, current_iteration:int, total_iteration:int):
        self._tqdm.update(current_iteration - self._tqdm.n)
        self._tqdm.close()
        self._tqdm = None
    
    def _progress_tqdm(self, current_iteration:int, total_iteration:int):
        self._tqdm.set_postfix_str(f'LLPW: {self.ll_per_word:.6f}')
        self._tqdm.update(current_iteration - self._tqdm.n)
    
    def _show_progress(self, current_iteration:int, total_iteration:int):
        if current_iteration == 0:
            self._init_tqdm(current_iteration, total_iteration)
        elif current_iteration == total_iteration:
            self._close_tqdm(current_iteration, total_iteration)
        else:
            self._progress_tqdm(current_iteration, total_iteration)
    
class DMRModel(_DMRModel, LDAModel):
    '''This type provides Dirichlet Multinomial Regression(DMR) topic model and its implementation is based on the following papers:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=0.0000000001, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    an initial value of exponential of mean of normal distribution for `lambdas`, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic - word
sigma : float
    standard deviation of normal distribution for `lambdas`
alpha_epsilon : float
    small smoothing value for preventing `exp(lambdas)` to be near zero
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            alpha,
            eta,
            sigma,
            alpha_epsilon,
            seed,
            corpus,
            transform,
        )

    def add_doc(self, words, metadata='', multi_metadata=[], ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with `metadata` and return an index of the inserted document.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
metadata : str
    metadata of the document (e.g., author, title or year)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, metadata, multi_metadata, ignore_empty_words)
    
    def make_doc(self, words, metadata='', multi_metadata=[]) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.models.LDAModel.infer` method.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
metadata : str
    metadata of the document (e.g., author, title or year)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
'''
        return self._make_doc(words, metadata, multi_metadata)
    
    def get_topic_prior(self, metadata='', multi_metadata=[], raw=False) -> List[float]:
        '''.. versionadded:: 0.12.0

Calculate the topic prior of any document with the given `metadata` and `multi_metadata`. 
If `raw` is true, the value without applying `exp()` is returned, otherwise, the value with applying `exp()` is returned.

The topic prior is calculated as follows:

`np.dot(lambda_[:, id(metadata)], np.concat([[1], multi_hot(multi_metadata)]))`

where `idx(metadata)` and `multi_hot(multi_metadata)` indicates 
an integer id of given `metadata` and multi-hot encoded binary vector for given `multi_metadata` respectively.


Parameters
----------
metadata : str
    metadata of the document (e.g., author, title or year)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
raw : bool
    If `raw` is true, the raw value of parameters without applying `exp()` is returned.
'''
        return self._get_topic_prior(metadata, multi_metadata, raw)
    
    @property
    def f(self) -> float:
        '''the number of metadata features (read-only)'''
        return self._f
    
    @property
    def sigma(self) -> float:
        '''the hyperparameter sigma (read-only)'''
        return self._sigma
    
    @property
    def alpha_epsilon(self) -> float:
        '''the smoothing value alpha-epsilon (read-only)'''
        return self._alpha_epsilon
    
    @property
    def metadata_dict(self):
        '''a dictionary of metadata in type `tomotopy.Dictionary` (read-only)'''
        return self._metadata_dict
    
    @property
    def multi_metadata_dict(self):
        '''a dictionary of metadata in type `tomotopy.Dictionary` (read-only)

.. versionadded:: 0.12.0

    This dictionary is distinct from `metadata_dict`.'''
        return self._multi_metadata_dict
    
    @property
    def lambdas(self) -> np.ndarray:
        '''parameter lambdas in the shape `[k, f]` (read-only)

.. warning::

    Prior to version 0.11.0, there was a bug in the lambda getter, so it yielded the wrong value. It is recommended to upgrade to version 0.11.0 or later.'''
        return self._lambdas
    
    @property
    def lambda_(self) -> np.ndarray:
        '''parameter lambdas in the shape `[k, len(metadata_dict), l]` where `k` is the number of topics and `l` is the size of vector for multi_metadata (read-only)

See `tomotopy.models.DMRModel.get_topic_prior` for the relation between the lambda parameter and the topic prior.

.. versionadded:: 0.12.0
'''
        return self._lambda_
    
    @property
    def alpha(self) -> np.ndarray:
        '''Dirichlet prior on the per-document topic distributions for each metadata in the shape `[k, f]`. Equivalent to `np.exp(DMRModel.lambdas)` (read-only)

.. versionadded:: 0.9.0

.. warning::

    Prior to version 0.11.0, there was a bug in the lambda getter, so it yielded the wrong value. It is recommended to upgrade to version 0.11.0 or later.'''
        return self._alpha
    
    def _summary_basic_info(self, file):
        LDAModel._summary_basic_info(self, file)
        md_cnt = Counter(doc.metadata for doc in self.docs)
        if len(md_cnt) > 1:
            print('| Metadata of docs and its distribution', file=file)
            for md in self.metadata_dict:
                print('|  {}: {}'.format(md, md_cnt.get(md, 0)), file=file)
        md_cnt = Counter()
        [md_cnt.update(doc.multi_metadata) for doc in self.docs]
        if len(md_cnt) > 0:
            print('| Multi-Metadata of docs and its distribution', file=file)
            for md in self.multi_metadata_dict:
                print('|  {}: {}'.format(md, md_cnt.get(md, 0)), file=file)

    def _summary_params_info(self, file):
        print('| lambda (feature vector per metadata of documents)\n'
            '|  {}'.format(_format_numpy(self.lambda_, '|  ')), file=file)
        print('| alpha (Dirichlet prior on the per-document topic distributions for each metadata)', file=file)
        for i, md in enumerate(self.metadata_dict):
            print('|  {}: {}'.format(md, _format_numpy(self.alpha[:, i], '|    ')), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)


class GDMRModel(_GDMRModel, DMRModel):
    '''This type provides Generalized DMR(g-DMR) topic model and its implementation is based on the following papers:

> * Lee, M., & Song, M. Incorporating citation impact into analysis of research trends. Scientometrics, 1-34.

.. versionadded:: 0.8.0

.. warning::

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.models.DMRModel`.
    So `metadata` arguments in the older codes should be replaced with `numeric_metadata` to work in version 0.11.0.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, degrees=[], alpha=0.1, eta=0.01, sigma=1.0, sigma0=3.0, decay=0, alpha_epsilon=0.0000000001, metadata_range=None, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
degrees : Iterable[int]
    a list of the degrees of Legendre polynomials for TDF(Topic Distribution Function). Its length should be equal to the number of metadata variables.

    Its default value is `[]` in which case the model doesn't use any metadata variable and as a result, it becomes the same as an LDA or DMR model. 
alpha : Union[float, Iterable[float]]
    exponential of mean of normal distribution for `lambdas`, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic - word
sigma : float
    standard deviation of normal distribution for non-constant terms of `lambdas`
sigma0 : float
    standard deviation of normal distribution for constant terms of `lambdas`
decay : float
    .. versionadded:: 0.11.0

    decay's exponent that causes the coefficient of the higher-order term of `lambdas` to become smaller
alpha_epsilon : float
    small smoothing value for preventing `exp(lambdas)` to be near zero
metadata_range : Iterable[Iterable[float]]
    a list of minimum and maximum value of each numeric metadata variable. Its length should be equal to the length of `degrees`.
    
    For example, `metadata_range = [(2000, 2017), (0, 1)]` means that the first variable has a range from 2000 and 2017 and the second one has a range from 0 to 1.
	Its default value is `None` in which case the ranges of each variable are obtained from input documents.
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    a list of documents to be added into the model
transform : Callable[dict, dict]
    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            degrees,
            alpha,
            eta,
            sigma,
            sigma0,
            decay,
            alpha_epsilon,
            metadata_range,
            seed,
            corpus,
            transform,
        )

    def add_doc(self, words, numeric_metadata=[], metadata='', multi_metadata=[], ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with `metadata` and return an index of the inserted document.

.. versionchanged:: 0.11.0

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.models.DMRModel`.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
numeric_metadata : Iterable[float]
    continuous numeric metadata variable of the document. Its length should be equal to the length of `degrees`.
metadata : str
    categorical metadata of the document (e.g., author, title, journal or country)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, numeric_metadata, metadata, multi_metadata, ignore_empty_words)
    
    def make_doc(self, words, numeric_metadata=[], metadata='', multi_metadata=[]) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.models.LDAModel.infer` method.

.. versionchanged:: 0.11.0

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.models.DMRModel`.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
numeric_metadata : Iterable[float]
    continuous numeric metadata variable of the document. Its length should be equal to the length of `degrees`.
metadata : str
    categorical metadata of the document (e.g., author, title, journal or country)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
'''
        return self._make_doc(words, numeric_metadata, metadata, multi_metadata)
    
    def tdf(self, numeric_metadata, metadata='', multi_metadata=[], normalize=True) -> List[float]:
        '''Calculate a topic distribution for given `numeric_metadata` value. It returns a list with length `k`.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
numeric_metadata : Iterable[float]
    continuous metadata variable whose length should be equal to the length of `degrees`.
metadata : str    
    categorical metadata variable
multi_metadata : Iterable[str]
    categorical metadata variables (for multiple values)
normalize : bool
    If true, the method returns probabilities for each topic in range [0, 1]. Otherwise, it returns raw values in logit.
'''
        return self._tdf(numeric_metadata, metadata, multi_metadata, normalize)
    
    def tdf_linspace(self, numeric_metadata_start, numeric_metadata_stop, num, metadata='', multi_metadata=[], endpoint=True, normalize=True) -> np.ndarray:
        '''Calculate topic distributions over a linspace of `numeric_metadata` values.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
numeric_metadata_start : Iterable[float]
    the starting value of each continuous metadata variable whose length should be equal to the length of `degrees`.
numeric_metadata_stop : Iterable[float]
    the end value of each continuous metadata variable whose length should be equal to the length of `degrees`.
num : Iterable[int]
    the number of samples to generate for each metadata variable. Must be non-negative. Its length should be equal to the length of `degrees`.
metadata : str
    categorical metadata variable
multi_metadata : Iterable[str]
    categorical metadata variables (for multiple values)
endpoint : bool
    If True, `metadata_stop` is the last sample. Otherwise, it is not included. Default is True.
normalize : bool
    If true, the method returns probabilities for each topic in range [0, 1]. Otherwise, it returns raw values in logit.

Returns
-------
samples : ndarray
    with shape `[*num, k]`. 
'''
        return self._tdf_linspace(numeric_metadata_start, numeric_metadata_stop, num, metadata, multi_metadata, endpoint, normalize)
    
    @property
    def degrees(self) -> List[int]:
        '''the degrees of Legendre polynomials (read-only)'''
        return self._degrees

    @property
    def sigma0(self) -> float:
        '''the hyperparameter sigma0 (read-only)'''
        return self._sigma0
    
    @property
    def decay(self) -> float:
        '''the hyperparameter decay (read-only)'''
        return self._decay
    
    @property
    def metadata_range(self) -> List[Tuple[float, float]]:
        '''the ranges of each metadata variable (read-only)'''
        return self._metadata_range
    
    def _summary_basic_info(self, file):
        LDAModel._summary_basic_info(self, file)

        md_cnt = Counter(doc.metadata for doc in self.docs)
        if len(md_cnt) > 1:
            print('| Categorical metadata of docs and its distribution', file=file)
            for md in self.metadata_dict:
                print('|  {}: {}'.format(md, md_cnt.get(md, 0)), file=file)
        md_cnt = Counter()
        [md_cnt.update(doc.multi_metadata) for doc in self.docs]
        if len(md_cnt) > 0:
            print('| Categorical multi-metadata of docs and its distribution', file=file)
            for md in self.multi_metadata_dict:
                print('|  {}: {}'.format(md, md_cnt.get(md, 0)), file=file)

        md_stack = np.stack([doc.numeric_metadata for doc in self.docs])
        md_min = md_stack.min(axis=0)
        md_max = md_stack.max(axis=0)
        md_avg = np.average(md_stack, axis=0)
        md_std = np.std(md_stack, axis=0)
        print('| Numeric metadata distribution of docs', file=file)
        for i in range(md_stack.shape[1]):
            print('|  #{}: Range={:.5}~{:.5}, Avg={:.5}, Stdev={:.5}'.format(i, md_min[i], md_max[i], md_avg[i], md_std[i]), file=file)

    def _summary_params_info(self, file):
        print('| lambda (feature vector per metadata of documents)\n'
            '|  {}'.format(_format_numpy(self.lambda_, '|  ')), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)


class HDPModel(_HDPModel, LDAModel):
    '''This type provides Hierarchical Dirichlet Process(HDP) topic model and its implementation is based on the following papers:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

.. versionchanged:: 0.3.0

    Since version 0.3.0, hyperparameter estimation for `alpha` and `gamma` has been added. You can turn off this estimation by setting `optim_interval` to zero.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, initial_k=2, alpha=0.1, eta=0.01, gamma=0.1, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
initial_k : int
    the initial number of topics between 2 ~ 32767
    The number of topics will be adjusted based on the data during training.
	
	Since version 0.3.0, the default value has been changed to 2 from 1.
alpha : float
    concentration coefficient of Dirichlet Process for document-table 
eta : float
    hyperparameter of Dirichlet distribution for topic-word
gamma : float
    concentration coefficient of Dirichlet Process for table-topic
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            initial_k,
            alpha,
            eta,
            gamma,
            seed,
            corpus,
            transform,
        )

    def is_live_topic(self, topic_id) -> bool:
        '''Return `True` if the topic `topic_id` is valid, otherwise return `False`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._is_live_topic(topic_id)
    
    def convert_to_lda(self, topic_threshold=0.0) -> Tuple['LDAModel', List[int]]:
        '''.. versionadded:: 0.8.0

Convert the current HDP model to equivalent LDA model and return `(new_lda_model, new_topic_id)`.
Topics with proportion less than `topic_threshold` are removed in `new_lda_model`.

`new_topic_id` is an array of length `HDPModel.k` and `new_topic_id[i]` indicates a topic id of new LDA model, equivalent to topic `i` of original HDP model.
If topic `i` of original HDP model is not alive or is removed in LDA model, `new_topic_id[i]` would be `-1`.

Parameters
----------
topic_threshold : float
    Topics with proportion less than this value is removed in new LDA model.
    The default value is 0, and it means no topic except not alive is removed.
'''
        return self._convert_to_lda(LDAModel, topic_threshold)
    
    def purge_dead_topics(self) -> List[int]:
        '''.. versionadded:: 0.12.3

Purge all non-alive topics from the model and return `new_topic_ids`. After called, `HDPModel.k` shrinks to `HDPModel.live_k` and all topics of the model become live.

`new_topic_id` is an array of length `HDPModel.k` and `new_topic_id[i]` indicates a topic id of the new model, equivalent to topic `i` of previous HDP model.
If topic `i` of previous HDP model is not alive or is removed in the new model, `new_topic_id[i]` would be `-1`.
'''
        return self._purge_dead_topics()
    
    @property
    def gamma(self) -> float:
        '''the hyperparameter gamma (read-only)'''
        return self._gamma
    
    @property
    def live_k(self) -> int:
        '''the number of alive topics (read-only)'''
        return self._live_k
    
    @property
    def num_tables(self) -> int:
        '''the number of total tables (read-only)'''
        return self._num_tables
    
    def _progress_tqdm(self, current_iteration:int, total_iteration:int):
        self._tqdm.set_postfix_str(f'# Topics: {self.live_k}, LLPW: {self.ll_per_word:.6f}')
        self._tqdm.update(current_iteration - self._tqdm.n)
    
    def _summary_params_info(self, file):
        print('| alpha (concentration coefficient of Dirichlet Process for document-table)\n'
            '|  {:.5}'.format(self.alpha), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)
        print('| gamma (concentration coefficient of Dirichlet Process for table-topic)\n'
            '|  {:.5}'.format(self.gamma), file=file)
        print('| Number of Topics: {}'.format(self.live_k), file=file)
        print('| Number of Tables: {}'.format(self.num_tables), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        for k in range(self.k):
            if not self.is_live_topic(k): continue
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('| #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

class MGLDAModel(_MGLDAModel, LDAModel):
    '''This type provides Multi Grain Latent Dirichlet Allocation(MG-LDA) topic model and its implementation is based on the following papers:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k_g : int
    the number of global topics between 1 ~ 32767
k_l : int
    the number of local topics between 1 ~ 32767
t : int
    the size of sentence window
alpha_g : float
    hyperparameter of Dirichlet distribution for document-global topic
alpha_l : float
    hyperparameter of Dirichlet distribution for document-local topic
alpha_mg : float
    hyperparameter of Dirichlet distribution for global-local selection (global coef)
alpha_ml : float
    hyperparameter of Dirichlet distribution for global-local selection (local coef)
eta_g : float
    hyperparameter of Dirichlet distribution for global topic-word
eta_l : float
    hyperparameter of Dirichlet distribution for local topic-word
gamma : float
    hyperparameter of Dirichlet distribution for sentence-window
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k_g,
            k_l,
            t,
            alpha_g,
            alpha_l,
            alpha_mg,
            alpha_ml,
            eta_g,
            eta_l,
            gamma,
            seed,
            corpus,
            transform,
        )
    
    def add_doc(self, words, delimiter='.', ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
delimiter : str
    a sentence separator. `words` will be separated by this value into sentences.
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, delimiter, ignore_empty_words)
    
    def make_doc(self, words, delimiter='.') -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
delimiter : str
    a sentence separator. `words` will be separated by this value into sentences.
'''
        return self._make_doc(words, delimiter)
    
    def get_topic_words(self, topic_id, top_n=10) -> List[Tuple[str, float]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int 
    A number in range [0, `k_g`) indicates a global topic and 
    a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
'''
        return self._get_topic_words(topic_id, top_n)
    
    def get_topic_word_dist(self, topic_id, normalize=True) -> List[float]:
        '''Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int 
    A number in range [0, `k_g`) indicates a global topic and 
    a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
'''
        return self._get_topic_word_dist(topic_id, normalize)
    
    @property
    def k_g(self) -> int:
        '''the hyperparameter k_g (read-only)'''
        return self._k_g
    
    @property
    def k_l(self) -> int:
        '''the hyperparameter k_l (read-only)'''
        return self._k_l
    
    @property
    def gamma(self) -> float:
        '''the hyperparameter gamma (read-only)'''
        return self._gamma
    
    @property
    def t(self) -> int:
        '''the hyperparameter t (read-only)'''
        return self._t
    
    @property
    def alpha_g(self) -> float:
        '''the hyperparameter alpha_g (read-only)'''
        return self._alpha
    
    @property
    def alpha_l(self) -> float:
        '''the hyperparameter alpha_l (read-only)'''
        return self._alpha_l
    
    @property
    def alpha_mg(self) -> float:
        '''the hyperparameter alpha_mg (read-only)'''
        return self._alpha_mg
    
    @property
    def alpha_ml(self) -> float:
        '''the hyperparameter alpha_ml (read-only)'''
        return self._alpha_ml
    
    @property
    def eta_g(self) -> float:
        '''the hyperparameter eta_g (read-only)'''
        return self._eta
    
    @property
    def eta_l(self) -> float:
        '''the hyperparameter eta_l (read-only)'''
        return self._eta_l

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        print('| Global Topic', file=file)
        for k in range(self.k):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('|  #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)
        print('| Local Topic', file=file)
        for k in range(self.k_l):
            words = ' '.join(w for w, _ in self.get_topic_words(k + self.k, top_n=topic_word_top_n))
            print('|  #{} ({}) : {}'.format(k, topic_cnt[k + self.k], words), file=file)

class PAModel(_PAModel, LDAModel):
    '''This type provides Pachinko Allocation(PA) topic model and its implementation is based on the following papers:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k1=1, k2=1, alpha=0.1, subalpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k1 : int
    the number of super topics between 1 ~ 32767
k2 : int
    the number of sub topics between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    initial hyperparameter of Dirichlet distribution for document-super topic, given as a single `float` in case of symmetric prior and as a list with length `k1` of `float` in case of asymmetric prior.
subalpha : Union[float, Iterable[float]]
    .. versionadded:: 0.11.0

    initial hyperparameter of Dirichlet distribution for super-sub topic, given as a single `float` in case of symmetric prior and as a list with length `k2` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for sub topic-word
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k1,
            k2,
            alpha,
            subalpha,
            eta,
            seed,
            corpus,
            transform,
        )

    def get_topic_words(self, sub_topic_id, top_n=10) -> List[Tuple[str, float]]:
        '''Return the `top_n` words and their probabilities in the sub topic `sub_topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
sub_topic_id : int
    indicating the sub topic, in range [0, `k2`)
'''
        return self._get_topic_words(sub_topic_id, top_n)
    
    def get_topic_word_dist(self, sub_topic_id, normalize=True) -> List[float]:
        '''Return the word distribution of the sub topic `sub_topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current sub topic.

Parameters
----------
sub_topic_id : int
    indicating the sub topic, in range [0, `k2`)
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
'''
        return self._get_topic_word_dist(sub_topic_id, normalize)
    
    def get_sub_topics(self, super_topic_id, top_n=10) -> List[Tuple[int, float]]:
        '''.. versionadded:: 0.1.4

Return the `top_n` sub topics and their probabilities in the super topic `super_topic_id`.
The return type is a `list` of (subtopic:`int`, probability:`float`).

Parameters
----------
super_topic_id : int
    indicating the super topic, in range [0, `k1`)
'''
        return self._get_sub_topics(super_topic_id, top_n)
    
    def get_sub_topic_dist(self, super_topic_id, normalize=True) -> List[float]:
        '''Return a distribution of the sub topics in a super topic `super_topic_id`.
The returned value is a `list` that has `k2` fraction numbers indicating probabilities for each sub topic in the current super topic.

Parameters
----------
super_topic_id : int
    indicating the super topic, in range [0, `k1`)
'''
        return self._get_sub_topic_dist(super_topic_id, normalize)
    
    def infer(self, doc, iterations=100, tolerance=-1, workers=0, parallel=0, together=False, transform=None) -> Tuple[Union[Tuple[List[float], List[float]], List[Tuple[List[float], List[float]]], Corpus], List[float]]:
        '''.. versionadded:: 0.5.0

Return the inferred topic distribution and sub-topic distribution from unseen `doc`s.

Parameters
----------
doc : Union[tomotopy.utils.Document, Iterable[tomotopy.utils.Document], tomotopy.utils.Corpus]
    an instance of `tomotopy.utils.Document` or a `list` of instances of `tomotopy.utils.Document` to be inferred by the model.
    It can be acquired from `tomotopy.models.LDAModel.make_doc` method.

    .. versionchanged:: 0.10.0

        Since version 0.10.0, `infer` can receive a raw corpus instance of `tomotopy.utils.Corpus`. 
        In this case, you don't need to call `make_doc`. `infer` would generate documents bound to the model, estimate its topic distributions and
        return a corpus containing generated documents as the result.
iterations : int
    an integer indicating the number of iteration to estimate the distribution of topics of `doc`.
    The higher value will generate a more accurate result.
tolerance : float
    This parameter is not currently used.
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for inference. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.
together : bool
    all `doc`s are inferred together in one process if True, otherwise each `doc` is inferred independently. Its default value is `False`.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    a callable object to manipulate arbitrary keyword arguments for a specific topic model. 
    Available when `doc` is given as an instance of `tomotopy.utils.Corpus`.

Returns
-------
result : Union[Tuple[List[float], List[float]], List[Tuple[List[float], List[float]]], tomotopy.utils.Corpus]
    If `doc` is given as a single `tomotopy.utils.Document`, `result` is a tuple of `List[float]` indicating its topic distribution and `List[float]` indicating its sub-topic distribution.
    
    If `doc` is given as a list of `tomotopy.utils.Document`s, `result` is a list of `List[float]` indicating topic distributions for each document.
    
    If `doc` is given as an instance of `tomotopy.utils.Corpus`, `result` is another instance of `tomotopy.utils.Corpus` which contains inferred documents.
    You can get topic distribution for each document using `tomotopy.utils.Document.get_topic_dist` and sub-topic distribution using `tomotopy.utils.Document.get_sub_topic_dist`
log_ll : List[float]
    a list of log-likelihoods for each `doc`
'''
        return self._infer(doc, iterations, tolerance, workers, parallel, together, transform)
    
    def get_count_by_super_topic(self) -> List[int]:
        '''Return the number of words allocated to each super-topic.

.. versionadded:: 0.9.0'''
        return self._get_count_by_super_topic()
    
    @property
    def k1(self) -> int:
        '''k1, the number of super topics (read-only)'''
        return self._k
    
    @property
    def k2(self) -> int:
        '''k2, the number of sub topics (read-only)'''
        return self._k2
    
    @property
    def alpha(self) -> float:
        '''Dirichlet prior on the per-document super topic distributions in shape `[k1]` (read-only)

.. versionadded:: 0.9.0'''
        return self._alpha
    
    @property
    def subalpha(self) -> float:
        '''Dirichlet prior on the sub topic distributions for each super topic in shape `[k1, k2]` (read-only)

.. versionadded:: 0.9.0'''
        return self._subalpha
    
    def _summary_params_info(self, file):
        print('| alpha (Dirichlet prior on the per-document super topic distributions)\n'
            '|  {}'.format(_format_numpy(self.alpha, '|  ')), file=file)
        print('| subalpha (Dirichlet prior on the sub topic distributions for each super topic)', file=file)
        for k1 in range(self.k1):
            print('|  Super #{}: {}'.format(k1, _format_numpy(self.subalpha[k1], '|   ')), file=file)
        print('| eta (Dirichlet prior on the per-subtopic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_super_topic()
        print('| Sub-topic distribution of Super-topics', file=file)
        for k in range(self.k1):
            words = ' '.join('#{}'.format(w) for w, _ in self.get_sub_topics(k, top_n=topic_word_top_n))
            print('|  #Super{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)
        topic_cnt = self.get_count_by_topics()
        print('| Word distribution of Sub-topics', file=file)
        for k in range(self.k2):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('|  #{} ({}) : {}'.format(k, topic_cnt[k], words), file=file)

class HPAModel(_HPAModel, PAModel):
    '''This type provides Hierarchical Pachinko Allocation(HPA) topic model and its implementation is based on the following papers:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k1=1, k2=1, alpha=0.1, subalpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k1 : int
    the number of super topics between 1 ~ 32767
k2 : int
    the number of sub topics between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    initial hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k1 + 1` of `float` in case of asymmetric prior.
subalpha : Union[float, Iterable[float]]
    .. versionadded:: 0.11.0

    initial hyperparameter of Dirichlet distribution for super-sub topic, given as a single `float` in case of symmetric prior and as a list with length `k2 + 1` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k1,
            k2,
            alpha,
            subalpha,
            eta,
            seed,
            corpus,
            transform,
        )

    def get_topic_words(self, topic_id, top_n=10) -> List[Tuple[str, float]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    0 indicates the top topic, 
    a number in range [1, 1 + `k1`) indicates a super topic and
    a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
'''
        return self._get_topic_words(topic_id, top_n)
    
    def get_topic_word_dist(self, topic_id, normalize=True) -> List[float]:
        '''Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int
    0 indicates the top topic, 
    a number in range [1, 1 + `k1`) indicates a super topic and
    a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
'''
        return self._get_topic_word_dist(topic_id, normalize)
    
    @property
    def alpha(self) -> float:
        '''Dirichlet prior on the per-document super topic distributions in shape `[k1 + 1]`. 
Its element 0 indicates the prior to the top topic and elements 1 ~ k1 indicates ones to the super topics. (read-only)

.. versionadded:: 0.9.0'''
        return self._alpha
    
    @property
    def subalpha(self) -> float:
        '''Dirichlet prior on the sub topic distributions for each super topic in shape `[k1, k2 + 1]`.
Its `[x, 0]` element indicates the prior to the super topic `x` 
and `[x, 1 ~ k2]` elements indicate ones to the sub topics in the super topic `x`. (read-only)

.. versionadded:: 0.9.0'''
        return self._subalpha
    
    def _summary_params_info(self, file):
        print('| alpha (Dirichlet prior on the per-document super topic distributions)\n'
            '|  {} {}'.format(self.alpha[:1], _format_numpy(self.alpha[1:], '|  ')), file=file)
        print('| subalpha (Dirichlet prior on the sub topic distributions for each super topic)', file=file)
        for k1 in range(self.k1):
            print('|  Super #{}: {} {}'.format(k1, self.subalpha[k1, :1], _format_numpy(self.subalpha[k1, 1:], '|   ')), file=file)
        print('| eta (Dirichlet prior on the per-subtopic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        words = ' '.join(w for w, _ in self.get_topic_words(0, top_n=topic_word_top_n))
        print('| Top-topic ({}) : {}'.format(topic_cnt[0], words), file=file)
        print('| Super-topics', file=file)
        for k in range(1, 1 + self.k1):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('|  #Super{} ({}) : {}'.format(k - 1, topic_cnt[k], words), file=file)
            words = ' '.join('#{}'.format(w) for w, _ in self.get_sub_topics(k - 1, top_n=topic_word_top_n))
            print('|    its sub-topics : {}'.format(words), file=file)
        print('| Sub-topics', file=file)
        for k in range(1 + self.k1, 1 + self.k1 + self.k2):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('|  #{} ({}) : {}'.format(k - 1 - self.k1, topic_cnt[k], words), file=file)

class CTModel(_CTModel, LDAModel):
    '''.. versionadded:: 0.2.0
This type provides Correlated Topic Model (CTM) and its implementation is based on the following papers:
	
> * Blei, D., & Lafferty, J. (2006). Correlated topic models. Advances in neural information processing systems, 18, 147.
> * Mimno, D., Wallach, H., & McCallum, A. (2008, December). Gibbs sampling for logistic normal topic models with graph-based priors. In NIPS Workshop on Analyzing Graphs (Vol. 61).'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, smoothing_alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
smoothing_alpha : Union[float, Iterable[float]]
    small smoothing value for preventing topic counts to be zero, given as a single `float` in case of symmetric and as a list with length `k` of `float` in case of asymmetric.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            smoothing_alpha,
            eta,
            seed,
            corpus,
            transform,
        )

    def get_correlations(self, topic_id=None) -> List[float]:
        '''Return correlations between the topic `topic_id` and other topics.
The returned value is a `list` of `float`s of size `tomotopy.models.LDAModel.k`.

Parameters
----------
topic_id : Union[int, None]
    an integer in range [0, `k`), indicating the topic
    
    If omitted, the whole correlation matrix is returned.
'''
        return self._get_correlations(topic_id)
    
    @property
    def num_beta_samples(self) -> int:
        '''the number of times to sample beta parameters, default value is 10.

CTModel samples `num_beta_samples` beta parameters for each document. 
The more beta it samples, the more accurate the distribution will be, but the more time it takes to learn. 
If you have a small number of documents in your model, keeping this value larger will help you get better result.
'''
        return self._num_beta_samples
    
    @num_beta_samples.setter
    def num_beta_samples(self, value: int):
        self._num_beta_samples = value
    
    @property
    def num_tmn_samples(self) -> int:
        '''the number of iterations for sampling Truncated Multivariate Normal distribution, default value is 5.

If your model shows biased topic correlations, increasing this value may be helpful.'''
        return self._num_tmn_samples
    
    @num_tmn_samples.setter
    def num_tmn_samples(self, value: int):
        self._num_tmn_samples = value

    @property
    def prior_mean(self) -> np.ndarray:
        '''the mean of prior logistic-normal distribution for the topic distribution (read-only)'''
        return self._prior_mean
    
    @property
    def prior_cov(self) -> np.ndarray:
        '''the covariance matrix of prior logistic-normal distribution for the topic distribution (read-only)'''
        return self._prior_cov
    
    @property
    def alpha(self) -> float:
        '''This property is not available in `CTModel`. Use `CTModel.prior_mean` and `CTModel.prior_cov` instead.

.. versionadded:: 0.9.1'''
        raise AttributeError("CTModel has no attribute 'alpha'. Use 'prior_mean' and 'prior_cov' instead.")
    
    def _summary_params_info(self, file):
        print('| prior_mean (Prior mean of Logit-normal for the per-document topic distributions)\n'
            '|  {}'.format(_format_numpy(self.prior_mean, '|  ')), file=file)
        print('| prior_cov (Prior covariance of Logit-normal for the per-document topic distributions)\n'
            '|  {}'.format(_format_numpy(self.prior_cov, '|  ')), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)    

class SLDAModel(_SLDAModel, LDAModel):
    '''This type provides supervised Latent Dirichlet Allocation(sLDA) topic model and its implementation is based on the following papers:
	
> * Mcauliffe, J. D., & Blei, D. M. (2008). Supervised topic models. In Advances in neural information processing systems (pp. 121-128).
> * Python version implementation using Gibbs sampling : https://github.com/Savvysherpa/slda

.. versionadded:: 0.2.0'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, vars='', alpha=0.1, eta=0.01, mu=[], nu_sq=[], glm_param=[], seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
vars : Iterable[str]
    indicating types of response variables.
    The length of `vars` determines the number of response variables, and each element of `vars` determines a type of the variable.
    The list of available types is like below:
    
    > * 'l': linear variable (any real value)
    > * 'b': binary variable (0 or 1)
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
mu : Union[float, Iterable[float]]
    mean of regression coefficients, default value is 0
nu_sq : Union[float, Iterable[float]]
    variance of regression coefficients, default value is 1
glm_param : Union[float, Iterable[float]]
    the parameter for Generalized Linear Model, default value is 1
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            vars,
            alpha,
            eta,
            mu,
            nu_sq,
            glm_param,
            seed,
            corpus,
            transform,
        )

    def add_doc(self, words, y=[], ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with response variables `y` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
y : Iterable[float]
    response variables of this document. 
    The length of `y` must be equal to the number of response variables of the model (`tomotopy.models.SLDAModel.f`).
    
    .. versionchanged:: 0.5.1
    
        If you have a missing value, you can set the item as `NaN`. Documents with `NaN` variables are included in modeling topics, but excluded from regression.
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, y, ignore_empty_words)
    
    def make_doc(self, words, y=[]) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and response variables `y` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
y : Iterable[float]
    response variables of this document. 
    The length of `y` doesn't have to be equal to the number of response variables of the model (`tomotopy.models.SLDAModel.f`).
    If the length of `y` is shorter than `tomotopy.models.SLDAModel.f`, missing values are automatically filled with `NaN`.
'''
        return self._make_doc(words, y)
    
    def get_regression_coef(self, var_id=None) -> List[float]:
        '''Return the regression coefficient of the response variable `var_id`.

Parameters
----------
var_id : int
    indicating the response variable, in range [0, `f`)

    If omitted, the whole regression coefficients with shape `[f, k]` are returned.
'''
        return self._get_regression_coef(var_id)
    
    def get_var_type(self, var_id) -> str:
        '''Return the type of the response variable `var_id`. 'l' means linear variable, 'b' means binary variable.'''
        return self._get_var_type(var_id)
    
    def estimate(self, doc) -> List[float]:
        '''Return the estimated response variable for `doc`.
If `doc` is an unseen document instance which is generated by `tomotopy.models.SLDAModel.make_doc` method, it should be inferred by `tomotopy.models.LDAModel.infer` method first.

Parameters
----------
doc : tomotopy.utils.Document
    an instance of document or a list of them to be used for estimating response variables
'''
        return self._estimate(doc)
    
    @property
    def f(self) -> int:
        '''the number of response variables (read-only)'''
        return self._f
    
    def _summary_initial_params_info_vars(self, v, file):
        var_type = {'l':'linear', 'b':'binary'}
        print('| vars: {}'.format(', '.join(map(var_type.__getitem__, v))), file=file)

    def _summary_params_info(self, file):
        LDAModel._summary_params_info(self, file)
        var_type = {'l':'linear', 'b':'binary'}
        print('| regression coefficients of response variables', file=file)
        for f in range(self.f):
            print('|  #{} ({}): {}'.format(f, 
                var_type.get(self.get_var_type(f)),
                _format_numpy(self.get_regression_coef(f), '|    ')
            ), file=file)


class LLDAModel(_LLDAModel, LDAModel):
    '''This type provides Labeled LDA(L-LDA) topic model and its implementation is based on the following papers:
	
> * Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.

.. versionadded:: 0.3.0

.. deprecated:: 0.11.0
    Use `tomotopy.models.PLDAModel` instead.'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            alpha,
            eta,
            seed,
            corpus,
            transform,
        )
    
    def add_doc(self, words, labels=[], ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with `labels` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
labels : Iterable[str]
    labels of the document
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, labels, ignore_empty_words)
    
    def make_doc(self, words, labels=[]) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and `labels` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
labels : Iterable[str]
    labels of the document
'''
        return self._make_doc(words, labels)
    
    def get_topic_words(self, topic_id, top_n=10, return_id=False) -> Union[List[Tuple[str, float]], List[Tuple[int, str, float]]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`) if `return_id` is False, or a `list` of (word_id:`int`, word:`str`, probability:`float`) if `return_id` is True.

Parameters
----------
topic_id : int
    Integers in the range [0, `l`), where `l` is the number of total labels, represent a topic that belongs to the corresponding label.
    The label name can be found by looking up `tomotopy.models.LLDAModel.topic_label_dict`.
    Integers in the range [`l`, `k`) represent a latent topic which does not belong to any label.
top_n : int
    the number of top words to return
return_id : bool
    If `True`, it returns a list of (word_id, word, probability) where `word_id` is an integer indicating the id of the word in the model's vocabulary. Otherwise, it returns a list of (word, probability).
'''
        return self._get_topic_words(topic_id, top_n, return_id)
    
    @property
    def topic_label_dict(self):
        '''a dictionary of topic labels in type `tomotopy.Dictionary` (read-only)'''
        return self._topic_label_dict
    
    def _summary_basic_info(self, file):
        LDAModel._summary_basic_info(self, file)
        label_cnt = Counter(l for doc in self.docs for l, _ in doc.labels)
        print('| Label of docs and its distribution', file=file)
        for lb in self.topic_label_dict:
            print('|  {}: {}'.format(lb, label_cnt.get(lb, 0)), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        for k in range(self.k):
            label = ('Label {} (#{})'.format(self.topic_label_dict[k], k) 
                if k < len(self.topic_label_dict) else '#{}'.format(k))
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('| {} ({}) : {}'.format(label, topic_cnt[k], words), file=file)

class PLDAModel(_PLDAModel, LDAModel):
    '''This type provides Partially Labeled LDA(PLDA) topic model and its implementation is based on the following papers:
	
> * Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.

.. versionadded:: 0.4.0'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, latent_topics=0, topics_per_label=1, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
latent_topics : int
    the number of latent topics, which are shared to all documents, between 1 ~ 32767
topics_per_label : int
    the number of topics per label between 1 ~ 32767
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            latent_topics,
            topics_per_label,
            alpha,
            eta,
            seed,
            corpus,
            transform,
        )

    def add_doc(self, words, labels=[], ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with `labels` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
labels : Iterable[str]
    labels of the document
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, labels, ignore_empty_words)
    
    def make_doc(self, words, labels=[]) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and `labels` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
labels : Iterable[str]
    labels of the document
'''
        return self._make_doc(words, labels)
    
    def get_topic_words(self, topic_id, top_n=10, return_id=False) -> Union[List[Tuple[str, float]], List[Tuple[int, str, float]]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    Integers in the range [0, `l` * `topics_per_label`), where `l` is the number of total labels, represent a topic that belongs to the corresponding label.
    The label name can be found by looking up `tomotopy.models.PLDAModel.topic_label_dict`.
    Integers in the range [`l` * `topics_per_label`, `l` * `topics_per_label` + `latent_topics`) represent a latent topic which does not belong to any label.
top_n : int
    the number of top words to return
return_id : bool
    If `True`, it returns a list of (word_id:`int`, word:`str`, probability:`float`) instead of (word:`str`, probability:`float`).
    
'''
        return self._get_topic_words(topic_id, top_n, return_id)
    
    @property
    def topic_label_dict(self):
        '''a dictionary of topic labels in type `tomotopy.Dictionary` (read-only)'''
        return self._topic_label_dict
    
    @property
    def latent_topics(self) -> int:
        '''the number of latent topics (read-only)'''
        return self._latent_topics
    
    @property
    def topics_per_label(self) -> int:
        '''the number of topics per label (read-only)'''
        return self._topics_per_label
    
    def _summary_basic_info(self, file):
        LDAModel._summary_basic_info(self, file)
        label_cnt = Counter(l for doc in self.docs for l, _ in doc.labels)
        print('| Label of docs and its distribution', file=file)
        for lb in self.topic_label_dict:
            print('|  {}: {}'.format(lb, label_cnt.get(lb, 0)), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        for k in range(self.k):
            l = k // self.topics_per_label
            label = ('Label {}-{} (#{})'.format(self.topic_label_dict[l], k % self.topics_per_label, k) 
                if l < len(self.topic_label_dict) else 'Latent {} (#{})'.format(k - self.topics_per_label * len(self.topic_label_dict), k))
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('| {} ({}) : {}'.format(label, topic_cnt[k], words), file=file)

class HLDAModel(_HLDAModel, LDAModel):
    '''This type provides Hierarchical LDA topic model and its implementation is based on the following papers:

> * Griffiths, T. L., Jordan, M. I., Tenenbaum, J. B., & Blei, D. M. (2004). Hierarchical topic models and the nested Chinese restaurant process. In Advances in neural information processing systems (pp. 17-24).

.. versionadded:: 0.4.0'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, depth=2, alpha=0.1, eta=0.01, gamma=0.1, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int    
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
depth : int
    the maximum depth level of hierarchy between 2 ~ 32767
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-depth level, given as a single `float` in case of symmetric prior and as a list with length `depth` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
gamma : float
    concentration coefficient of Dirichlet Process
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            depth,
            alpha,
            eta,
            gamma,
            seed,
            corpus,
            transform,
        )

    def is_live_topic(self, topic_id) -> bool:
        '''Return `True` if the topic `topic_id` is alive, otherwise return `False`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._is_live_topic(topic_id)
    
    def num_docs_of_topic(self, topic_id) -> int:
        '''Return the number of documents belonging to a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._num_docs_of_topic(topic_id)
    
    def level(self, topic_id) -> int:
        '''Return the level of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._level(topic_id)
    
    def parent_topic(self, topic_id) -> int:
        '''Return the topic ID of parent of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._parent_topic(topic_id)
    
    def children_topics(self, topic_id) -> List[int]:
        '''Return a list of topic IDs with children of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
'''
        return self._children_topics(topic_id)
    
    @property
    def gamma(self) -> float:
        '''the hyperparameter gamma (read-only)'''
        return self._gamma
    
    @property
    def live_k(self) -> int:
        '''the number of alive topics (read-only)'''
        return self._live_k
    
    @property
    def depth(self) -> int:
        '''the maximum depth level of hierarchy (read-only)'''
        return self._depth
    
    def _progress_tqdm(self, current_iteration:int, total_iteration:int):
        self._tqdm.set_postfix_str(f'# Topics: {self.live_k}, LLPW: {self.ll_per_word:.6f}')
        self._tqdm.update(current_iteration - self._tqdm.n)
    
    def _summary_params_info(self, file):
        print('| alpha (Dirichlet prior on the per-document depth level distributions)\n'
            '|  {}'.format(_format_numpy(self.alpha, '|  ')), file=file)
        print('| eta (Dirichlet prior on the per-topic word distribution)\n'
            '|  {:.5}'.format(self.eta), file=file)
        print('| gamma (concentration coefficient of Dirichlet Process)\n'
            '|  {:.5}'.format(self.gamma), file=file)
        print('| Number of Topics: {}'.format(self.live_k), file=file)

    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()

        def print_hierarchical(k=0, level=0):
            words = ' '.join(w for w, _ in self.get_topic_words(k, top_n=topic_word_top_n))
            print('| {}#{} ({}, {}) : {}'.format('  ' * level, k, topic_cnt[k], self.num_docs_of_topic(k), words), file=file)
            for c in np.sort(self.children_topics(k)):
                print_hierarchical(c, level + 1)

        print_hierarchical()

class DTModel(_DTModel, LDAModel):
    '''This type provides Dynamic Topic model and its implementation is based on the following papers:

> * Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).
> * Bhadury, A., Chen, J., Zhu, J., & Liu, S. (2016, April). Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web (pp. 381-390).
> https://github.com/Arnie0426/FastDTM

.. versionadded:: 0.7.0'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, t=1, alpha_var=0.1, eta_var=0.1, phi_var=0.1, lr_a=0.01, lr_b=0.1, lr_c=0.55, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
t : int
    the number of timepoints
alpha_var : float
    transition variance of alpha (per-document topic distribution)
eta_var : float
    variance of eta (topic distribution of each document) from its alpha 
phi_var : float
    transition variance of phi (word distribution of each topic)
lr_a : float
    shape parameter `a` greater than zero, for SGLD step size calculated as `e_i = a * (b + i) ^ (-c)`
lr_b : float
    shape parameter `b` greater than or equal to zero, for SGLD step size calculated as `e_i = a * (b + i) ^ (-c)`
lr_c : float
    shape parameter `c` with range (0.5, 1], for SGLD step size calculated as `e_i = a * (b + i) ^ (-c)`
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    a list of documents to be added into the model
transform : Callable[dict, dict]
    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            t,
            alpha_var,
            eta_var,
            phi_var,
            lr_a,
            lr_b,
            lr_c,
            seed,
            corpus,
            transform,
        )

    def add_doc(self, words, timepoint=0, ignore_empty_words=True) -> Optional[int]:
        '''Add a new document into the model instance with `timepoint` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
timepoint : int
    an integer with range [0, `t`)
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise an exception and makes the method return None.
'''
        return self._add_doc(words, timepoint, ignore_empty_words)
    
    def make_doc(self, words, timepoint=0) -> Document:
        '''Return a new `tomotopy.utils.Document` instance for an unseen document with `words` and `timepoint` that can be used for `tomotopy.models.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
timepoint : int
    an integer with range [0, `t`)
'''
        return self._make_doc(words, timepoint)
    
    def get_alpha(self, timepoint) -> List[float]:
        '''Return a `list` of alpha parameters for `timepoint`.

Parameters
----------
timepoint : int
    an integer with range [0, `t`)
'''
        return self._get_alpha(timepoint)
    
    def get_phi(self, timepoint, topic_id) -> List[float]:
        '''Return a `list` of phi parameters for `timepoint` and `topic_id`.

Parameters
----------
timepoint : int
    an integer with range [0, `t`)
topic_id : int
    an integer with range [0, `k`)
'''
        return self._get_phi(timepoint, topic_id)
    
    def get_topic_words(self, topic_id, timepoint, top_n=10) -> List[Tuple[str, float]]:
        '''Return the `top_n` words and their probabilities in the topic `topic_id` with `timepoint`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    an integer in range [0, `k`), indicating the topic
timepoint : int
	an integer in range [0, `t`), indicating the timepoint
'''
        return self._get_topic_words(topic_id, timepoint, top_n)
    
    def get_topic_word_dist(self, topic_id, timepoint, normalize=True) -> List[float]:
        '''Return the word distribution of the topic `topic_id` with `timepoint`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
timepoint : int
	an integer in range [0, `t`), indicating the timepoint
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
'''
        return self._get_topic_word_dist(topic_id, timepoint, normalize)
    
    def get_count_by_topics(self) -> np.ndarray:
        '''Return the number of words allocated to each timepoint and topic in the shape `[num_timepoints, k]`.

.. versionadded:: 0.9.0'''
        return self._get_count_by_topics()
    
    @property
    def lr_a(self) -> float:
        '''the shape parameter `a` greater than zero for SGLD step size (e_i = a * (b + i) ^ -c)'''
        return self._lr_a
    
    @lr_a.setter
    def lr_a(self, value: float):
        self._lr_a = value

    @property
    def lr_b(self) -> float:
        '''the shape parameter `b` greater than or equal to zero for SGLD step size (e_i = a * (b + i) ^ -c)'''
        return self._lr_b
    
    @lr_b.setter
    def lr_b(self, value: float):
        self._lr_b = value

    @property
    def lr_c(self) -> float:
        '''the shape parameter `c` with range (0.5, 1] for SGLD step size (e_i = a * (b + i) ^ -c)'''
        return self._lr_c
    
    @lr_c.setter
    def lr_c(self, value: float):
        self._lr_c = value

    @property
    def num_timepoints(self) -> int:
        '''the number of timepoints of the model (read-only)'''
        return self._num_timepoints
    
    @property
    def num_docs_by_timepoint(self) -> List[int]:
        '''the number of documents in the model by timepoint (read-only)'''
        return self._num_docs_by_timepoint
    
    @property
    def alpha(self) -> float:
        '''per-document topic distribution in the shape `[num_timepoints, k]` (read-only)

.. versionadded:: 0.9.0'''
        return self._alpha
    
    @property
    def eta(self):
        '''This property is not available in `DTModel`. Use `DTModel.docs[x].eta` instead.

.. versionadded:: 0.9.0'''
        raise AttributeError("DTModel has no attribute 'eta'. Use 'docs[x].eta' instead.")
    
    def _summary_params_info(self, file):
        print('| alpha (Dirichlet prior on the per-document topic distributions for each timepoint)\n'
            '|  {}'.format(_format_numpy(self.alpha, '|  ')), file=file)
        print('| phi (Dirichlet prior on the per-time&topic word distribution)\n'
            '|  ...', file=file)
        
    def _summary_topics_info(self, file, topic_word_top_n):
        topic_cnt = self.get_count_by_topics()
        for k in range(self.k):
            print('| #{} ({})'.format(k, topic_cnt[:, k].sum()), file=file)
            for t in range(self.num_timepoints):
                words = ' '.join(w for w, _ in self.get_topic_words(k, t, top_n=topic_word_top_n))
                print('|  t={} ({}) : {}'.format(t, topic_cnt[t, k], words), file=file)

class PTModel(_PTModel, LDAModel):
    '''.. versionadded:: 0.11.0
This type provides Pseudo-document based Topic Model (PTM) and its implementation is based on the following papers:
	
> * Zuo, Y., Wu, J., Zhang, H., Lin, H., Wang, F., Xu, K., & Xiong, H. (2016, August). Topic modeling of short texts: A pseudo-document view. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 2105-2114).'''

    def __init__(self,
                 tw='one', min_cf=0, min_df=0, rm_top=0, k=1, p=None, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None):
        '''Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded.
rm_top : int
    the number of top words to be removed. If you want to remove too common words from the model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
p : int
    the number of pseudo documents
    ..versionchanged:: 0.12.2
        The default value is changed to `10 * k`.
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-topic, given as a single `float` in case of symmetric prior and as a list with length `k` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
seed : int
    random seed. The default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    a list of documents to be added into the model
transform : Callable[dict, dict]
    a callable object to manipulate arbitrary keyword arguments for a specific topic model

'''
        # get initial params
        self.init_params = deepcopy({k: v for k, v in locals().items() if k != 'self' and not k.startswith('_')})
        self.init_params['version'] = __version__
        tw = _convert_term_weight(tw)

        super().__init__(
            tw,
            min_cf,
            min_df,
            rm_top,
            k,
            p,
            alpha,
            eta,
            seed,
            corpus,
            transform,
        )
    
    @property
    def p(self) -> int:
        '''the number of pseudo documents (read-only)

.. versionadded:: 0.11.0'''
        return self._p
