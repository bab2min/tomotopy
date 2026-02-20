'''
Submodule `tomotopy.utils` provides various utilities for topic modeling.
`tomotopy.utils.Corpus` class helps manage multiple documents easily.
The documents inserted into `Corpus` can be used with any topic model, and you can save the corpus preprocessed into a file and load the corpus from a file.
'''

from typing import Optional, List, Tuple

import re

from _tomotopy import (_Document, _UtilsCorpus, _UtilsVocabDict)

class Document(_Document):
    '''This type provides an abstract interface for accessing documents used in topic models.

An instance of this type can be acquired from `tomotopy.models.LDAModel.make_doc` method or `tomotopy.models.LDAModel.docs` member of each Topic Model instance.'''
    
    def get_topics(self, top_n=10, from_pseudo_doc=False) -> List[Tuple[int, float]]:
        '''Return the `top_n` topics with their probabilities for the document.

Parameters
----------
top_n : int
    the `n` in "top-n"
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    If True, it returns the topic distribution of its pseudo document. Only valid for `tomotopy.models.PTModel`.'''
        return super().get_topics(top_n, from_pseudo_doc)
    
    def get_topic_dist(self, normalize=True, from_pseudo_doc=False) -> List[float]:
        '''Return a distribution of the topics in the document.

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    If True, it returns the topic distribution of its pseudo document. Only valid for `tomotopy.models.PTModel`.'''
        return super().get_topic_dist(normalize, from_pseudo_doc)
    
    def get_sub_topics(self, top_n=10) -> List[Tuple[int, float]]:
        '''.. versionadded:: 0.5.0

Return the `top_n` sub topics with their probabilities for the document. (for only `tomotopy.models.PAModel`)'''
        return super().get_sub_topics(top_n)
    
    def get_sub_topic_dist(self, normalize=True) -> List[float]:
        '''.. versionadded:: 0.5.0

Return a distribution of the sub topics in the document. (for only `tomotopy.models.PAModel`)

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.'''
        return super().get_sub_topic_dist(normalize)
    
    def get_words(self, top_n=10) -> List[Tuple[int, float]]:
        '''.. versionadded:: 0.4.2

Return the `top_n` words with their probabilities for the document.'''
        return super().get_words(top_n)
    
    def get_count_vector(self) -> List[int]:
        '''.. versionadded:: 0.7.0

Return a count vector for the current document.'''
        return super().get_count_vector()
    
    def get_ll(self) -> float:
        '''.. versionadded:: 0.10.0

Return total log-likelihood for the current document.'''
        return super().get_ll()
    
    @property
    def words(self) -> List[int]:
        '''a `list` of IDs for each word (read-only)'''
        return super()._words
    
    @property
    def weights(self) -> List[float]:
        '''a `list` of weights for each word (read-only)'''
        return super()._weights
    
    @property
    def topics(self) -> List[int]:
        '''a `list` of topics for each word (read-only)

This represents super topics in `tomotopy.models.PAModel` and `tomotopy.models.HPAModel` model.'''
        return super()._topics
    
    @property
    def uid(self) -> str:
        '''a unique ID for the document (read-only)'''
        return super()._uid
    
    @property
    def metadata(self) -> str:
        '''categorical metadata of the document (for only `tomotopy.models.DMRModel` and `tomotopy.models.GDMRModel` model, read-only)'''
        return super()._metadata
    
    @property
    def multi_metadata(self) -> List[str]:
        '''categorical multiple metadata of the document (for only `tomotopy.models.DMRModel` and `tomotopy.models.GDMRModel` model, read-only)

.. versionadded:: 0.12.0'''
        return super()._multi_metadata
    
    @property
    def numeric_metadata(self) -> List[float]:
        '''continuous numeric metadata of the document (for only `tomotopy.models.GDMRModel` model, read-only)

.. versionadded:: 0.11.0'''
        return super()._numeric_metadata
    
    @property
    def subtopics(self) -> List[int]:
        '''a `list` of sub topics for each word (for only `tomotopy.models.PAModel` and `tomotopy.models.HPAModel` model, read-only)'''
        return super()._subtopics
    
    @property
    def windows(self) -> List[int]:
        '''a `list` of window IDs for each word (for only `tomotopy.models.MGLDAModel` model, read-only)'''
        return super()._windows
    
    @property
    def paths(self) -> List[int]:
        '''a `list` of topic ids by depth for a given document (for only `tomotopy.models.HLDAModel` model, read-only)

.. versionadded:: 0.7.1'''
        return super()._paths
    
    @property
    def beta(self) -> List[float]:
        '''a `list` of beta parameters for each topic (for only `tomotopy.models.CTModel` model, read-only)

.. versionadded:: 0.2.0'''
        return super()._beta
    
    @property
    def vars(self) -> List[float]:
        '''a `list` of response variables (for only `tomotopy.models.SLDAModel` model, read-only)

.. versionadded:: 0.2.0'''
        return super()._vars
    
    @property
    def labels(self) -> List[Tuple[str, List[float]]]:
        '''a `list` of (label, list of probabilities of each topic belonging to the label) of the document (for only `tomotopy.models.LLDAModel` and `tomotopy.models.PLDAModel` models, read-only)

.. versionadded:: 0.3.0'''
        return super()._labels
    
    @property
    def eta(self) -> List[float]:
        '''a `list` of eta parameters(topic distribution) for the current document (for only `tomotopy.models.DTModel` model, read-only)

.. versionadded:: 0.7.0'''
        return super()._eta
    
    @property
    def timepoint(self) -> int:
        '''a timepoint of the document (for only `tomotopy.models.DTModel` model, read-only)

.. versionadded:: 0.7.0'''
        return super()._timepoint
    
    @property
    def raw(self) -> Optional[str]:
        '''a raw text of the document (read-only)'''
        return super()._raw
        
    @property
    def span(self) -> List[Tuple[int, int]]:
        '''a span (tuple of a start position and an end position in bytes) for each word token in the document (read-only)'''
        return super()._span
    
    @property
    def pseudo_doc_id(self) -> int:
        '''an ID of a pseudo document where the document is allocated to (for only `tomotopy.models.PTModel` model, read-only)

.. versionadded:: 0.11.0'''
        return super()._pseudo_doc_id

class Corpus(_UtilsCorpus):
    '''`Corpus` class is a utility that makes it easy to manage large amounts of documents.
    An instance of `Corpus` can contain multiple preprocessed documents, and can be used directly by passing them as parameters of the topic modeling classes.
    '''
    class _VocabDict(_UtilsVocabDict):
        pass

    def __init__(self, tokenizer=None, batch_size=64, stopwords=None):
        '''Parameters
----------
tokenizer : Union[Callable[[str, Any], List[Union[str, Tuple[str, int, int]]]], Callable[[Iterable[Tuple[str, Any]]], Iterable[List[Union[str, Tuple[str, int, int]]]]]]
    a callable object for tokenizing raw documents. If `tokenizer` is provided, you can use `tomotopy.utils.Corpus.add_doc` method with `raw` and `user_data` parameters.
    `tokenizer` receives two arguments `raw` and `user_data` and 
    it should return an iterable of `str`(the tokenized word) or of Tuple[`str`, `int`, `int`] (the tokenized word, starting position of the word, the length of the word).
batch_size : int
    `tomotopy.utils.Corpus.process` method reads a bunch of documents and send them to `tomotopy.utils.Corpus.add_doc`. `batch_size` indicates the size of each batch.
stopwords : Union[Iterable[str], Callable[str, bool]]
    When calling `tomotopy.utils.Corpus.add_doc`, words in `stopwords` are not added to the document but are excluded.
    If `stopwords` is callable, a word is excluded from the document when `stopwords(word) == True`.
        '''
        super().__init__(self._VocabDict, None)
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        if callable(stopwords):
            self._stopwords = stopwords
        elif stopwords is None:
            self._stopwords = None
        else:
            self._stopwords = lambda x: x in set(stopwords)

    def _select_args_for_model(self, model_type:type, args:dict):
        import tomotopy as tp
        if model_type in (tp.DMRModel, tp.GDMRModel):
            return {k:v for k, v in args.items() if k in ('metadata')}
        if model_type in (tp.LLDAModel, tp.PLDAModel):
            return {k:v for k, v in args.items() if k in ('labels')}
        if model_type is tp.MGLDAModel:
            return {k:v for k, v in args.items() if k in ('delimiter')}
        if model_type is tp.SLDAModel:
            return {k:v for k, v in args.items() if k in ('y')}
        if model_type is tp.DTModel:
            return {k:v for k, v in args.items() if k in ('timepoint')}
        return {}    

    def add_doc(self, words=None, raw=None, user_data=None, **kargs) -> int:
        '''Add a new document into the corpus and return an index of the inserted document. 
This method requires either `words` parameter or `raw` and `user_data` parameters. 
If `words` parameter is provided, `words` are expected to be already preprocessed results.
If `raw` parameter is provided, `raw` is expected to be a raw string of document which isn't preprocessed yet, and `tokenizer` will be called for preprocessing the raw document.

If you need additional parameters for a specific topic model, such as `metadata` for `tomotopy.models.DMRModel` or `y` for `tomotopy.models.SLDAModel`, you can pass it as an arbitrary keyword argument.

Parameters
----------
words : Iterable[str]
    a list of words that are already preprocessed
raw : str
    a raw string of document which isn't preprocessed yet. 
    The `raw` parameter can be used only when the `tokenizer` parameter of `__init__` is set.
user_data : Any
    user data for `tokenizer`. The `raw` and `user_data` parameter are sent to `tokenizer`.
**kargs
    arbitrary keyword arguments for specific topic models
        '''
        return super().add_doc(words, raw, user_data, kargs)

    def process(self, data_feeder, show_progress=False, total=None) -> int:
        '''Add multiple documents into the corpus through a given iterator `data_feeder` and return the number of documents inserted.

Parameters
----------
data_feeder : Iterable[Union[str, Tuple[str, Any], Tuple[str, Any, dict]]]
    any iterable yielding a str `raw`, a tuple of (`raw`, `user_data`) or a tuple of (`raw`, `user_data`, `arbitrary_keyword_args`). 
        '''
        if self._tokenizer is None:
            raise ValueError("`tokenizer` must be set when using `tomotopy.utils.Corpus.process`")
        
        num = [0]
        raw_list = []
        metadata_list = []
        if show_progress:
            from tqdm import tqdm
            data_feeder_iter = iter(tqdm(data_feeder, total=total))
        else:
            data_feeder_iter = iter(data_feeder)
        def _generate():
            for _, d in zip(range(self._batch_size), data_feeder_iter):
                num[0] += 1
                if isinstance(d, tuple) and len(d) == 2:
                    raw_list.append(d[0])
                    metadata_list.append({})
                    yield d
                elif isinstance(d, tuple) and len(d) == 3:
                    raw_list.append(d[0])
                    metadata_list.append(d[2])
                    yield d[:2]
                elif isinstance(d, str):
                    raw_list.append(d)
                    metadata_list.append({})
                    yield (d, None)
                else:
                    raise ValueError("`data_feeder` must return an iterable of str, of Tuple[str, Any] or Tuple[str, Any, dict]")    
        
        while 1:
            added = super().add_docs(self._tokenizer(_generate()), iter(raw_list), iter(metadata_list))
            if added == 0: break
            raw_list.clear()
            metadata_list.clear()
        return num[0]

    def save(self, filename:str, protocol=0) -> None:
        '''Save the current instance into the file `filename`. 

Parameters
----------
filename : str
    a path for the file where the instance is saved
        '''
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename:str) -> 'Corpus':
        '''Load and return an instance from the file `filename`

Parameters
----------
filename : str
    a path for the file to be loaded
        '''
        import pickle
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        obj._stopwords = None
        return obj

    def __len__(self) -> int:
        return super().__len__()
    
    def extract_ngrams(self, min_cf=10, min_df=5, max_len=5, max_cand=5000, min_score=float('-inf'), normalized=False, workers=0) -> List:
        '''..versionadded:: 0.10.0

Extract frequent n-grams using PMI score

Parameters
----------
min_cf : int
    Minimum collection frequency of n-grams to be extracted
min_df : int
    Minimum document frequency of n-grams to be extracted
max_len : int
    Maximum length of n-grams to be extracted
max_cand : int
    Maximum number of n-grams to be extracted
min_score : float
    Minimum PMI score of n-grams to be extracted
normalized : bool
    whether to use Normalized PMI or just PMI
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.

Returns
-------
candidates : List[tomotopy.label.Candidate]
    The extracted n-gram candidates in `tomotopy.label.Candidate` type
        '''
        return super().extract_ngrams(min_cf, min_df, max_len, max_cand, min_score, normalized, workers)
    
    def concat_ngrams(self, cands, delimiter='_') -> None:
        '''..versionadded:: 0.10.0

Concatenate n-grams matching the given candidates in the corpus into single words

Parameters
----------
cands : Iterable[tomotopy.label.Candidate]
    n-gram candidates to be concatenated. It can be generated by `tomotopy.utils.Corpus.extract_ngrams`.
delimiter : str
    Delimiter to be used for concatenating words. Default value is `'_'`.
        '''
        return super().concat_ngrams(cands, delimiter)

class SimpleTokenizer:
    '''`SimpleTokenizer` provides a simple word-tokenizing utility with an arbitrary stemmer.'''
    def __init__(self, 
                 stemmer = None, 
                 pattern:str = None, 
                 lowercase = True, 
                 ngram_list:Optional[List[str]] = None,
                 ngram_delimiter:str = '_',
                 ):
        '''Parameters
----------
stemmer : Callable[str, str]
    a callable object for stemming words. If this value is set to `None`, words are not stemmed.
pattern : str
    a regex pattern for extracting tokens
lowercase : bool
    converts the token into lowercase if this is True

Here is an example of using SimpleTokenizer with NLTK for stemming.

.. include:: ./auto_labeling_code_with_porter.rst
'''
        self._pat = re.compile(pattern or r"""[^\s.,;:'"?!<>(){}\[\]\\/`~@#$%^&*|]+""")
        if stemmer and not callable(stemmer):
            raise ValueError("`stemmer` must be callable.")
        self._stemmer = stemmer or None
        self._lowercase = lowercase
        self._ngram_pat = None
        self._ngram_delimiter = ngram_delimiter
        if ngram_list:
            self.build_ngram_pat(ngram_list)

    def build_ngram_pat(self, ngram_list:List[str]):
        ngram_vocab = {}
        patterns = []

        for ngram in ngram_list:
            if self._lowercase:
                ngram = ngram.lower()
            words = self._pat.findall(ngram)
            if len(words) < 2:
                continue
            chrs = []
            for word in words:
                if self._stemmer is not None:
                    word = self._stemmer(word)
                try:
                    wid = ngram_vocab[word]
                except KeyError:
                    wid = chr(len(ngram_vocab) + 256)
                    ngram_vocab[word] = wid
                chrs.append(wid)
            patterns.append(''.join(chrs))
        
        if patterns:
            self._ngram_pat = re.compile('|'.join(sorted(patterns, key=lambda x: len(x), reverse=True)))
            self._ngram_vocab = ngram_vocab

    def _tokenize(self, raw:str):
        if self._ngram_pat is None:
            for g in self._pat.finditer(raw):
                start, end = g.span()
                word = g.group()
                if self._lowercase: 
                    word = word.lower()
                if self._stemmer is not None:
                    word = self._stemmer(word)
                yield word, start, end - start
        else:
            all_words = []
            all_spans = []
            chrs = []
            for g in self._pat.finditer(raw):
                all_spans.append(g.span())
                word = g.group()
                if self._lowercase: 
                    word = word.lower()
                if self._stemmer is not None:
                    word = self._stemmer(word)
                all_words.append(word)
                try:
                    chrs.append(self._ngram_vocab[word])
                except KeyError:
                    chrs.append(' ')
            chrs = ''.join(chrs)
            for g in self._ngram_pat.finditer(chrs):
                s, e = g.span()
                is_space = all(raw[ns:ne].isspace() for (_, ns), (ne, _) in zip(all_spans[s:e-1], all_spans[s+1:e]))
                if not is_space:
                    continue
                all_words[s] = self._ngram_delimiter.join(all_words[s:e])
                all_words[s+1:e] = [None] * (e - s - 1)
                all_spans[s] = (all_spans[s][0], all_spans[e-1][1])

            for (s, e), word in zip(all_spans, all_words):
                if word is None: continue
                yield word, s, e - s

    def __call__(self, raw:str, user_data=None):
        is_iterable = False
        # test raw is iterable
        if user_data is None and not isinstance(raw, str):
            try:
                iter(raw)
                is_iterable = True
            except TypeError:
                pass
        if is_iterable:
            for r, _ in raw:
                yield list(self._tokenize(r))
        else:
            yield from self._tokenize(raw)

