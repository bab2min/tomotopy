'''
Submodule `tomotopy.utils` provides various utilities for topic modeling. 
`tomotopy.utils.Corpus` class helps manage multiple documents easily. 
The documents inserted into `Corpus` can be used any topic models, and you can save the corpus preprocessed into a file and load the corpus from a file.
'''

class Corpus:
    '''`Corpus` class is a utility that makes it easy to manage large amounts of documents.
    An instance of `Corpus` can contain multiple preprocessed documents, and can be used directly by passing them as parameters of the topic modeling classes.
    '''

    class _VocabDict:
        def __init__(self):
            self.id2word = []
            self.word2id = {}
        
        def to_id(self, word):
            r = self.word2id.get(word, None)
            if not r is None: return r
            r = len(self.word2id)
            self.word2id[word] = r
            self.id2word.append(word)
            return r

        def to_word(self, id_):
            return self.id2word[id_]

    def __init__(self, tokenizer=None, batch_size=64, stopwords=None):
        '''Parameters
----------
tokenizer : Callable[[str, Any], Iterable[Union[str, Tuple[str, int, int]]]]
    a callable object for tokenizing raw documents. If `tokenizer` is provided, you can use `tomotopy.utils.Corpus.add_doc` method with `raw` and `user_data` parameters.
    `tokenizer` receives two arguments `raw` and `user_data` and 
    it should return an iterable of `str`(the tokenized word) or of Tuple[`str`, `int`, `int`] (the tokenized word, starting position of the word, the length of the word).
batch_size : int
    `tomotopy.utils.Corpus.process` method reads a bunch of documents and send them to `tomotopy.utils.Corpus.add_doc`. `batch_size` indicates the size of the bunch.
stopwords : Union[Iterable[str], Callable[str, bool]]
    When calling `tomotopy.utils.Corpus.add_doc`, words in `stopwords` are not added to the document but are excluded.
    If `stopwords` is callable, a word is excluded from the document when `stopwords(word) == True`.
        '''
        self._docs = []
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._vocab = Corpus._VocabDict()
        if callable(stopwords):
            self._stopwords = stopwords
        elif stopwords is None:
            self._stopwords = lambda x: False
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
    
    def _feed_docs_to(self, model, transform=None):
        if not self._docs: 
            raise ValueError("Cannot feed zero-size corpus.")
        
        model._update_vocab(self._vocab.id2word)
        transform = transform or (lambda x:x)

        if self._tokenizer:
            for doc in self._docs:
                model._add_doc(doc[0], raw=doc[1], start_pos=doc[2], length=doc[3], **self._select_args_for_model(type(model), transform(doc[4])))
        else:
            for doc in self._docs:
                model._add_doc(doc[0], **self._select_args_for_model(type(model), transform(doc[1])))

    def _tokenize(self, raw, user_data=None):
        tokens, ss, ls = [], [], []
        for t in self._tokenizer(raw, user_data=user_data):
            if type(t) is str:
                if self._stopwords(t): continue
                tokens.append(self._vocab.to_id(t))
            elif type(t) is tuple and len(t) == 3:
                if self._stopwords(t[0]): continue
                tokens.append(self._vocab.to_id(t[0]))
                ss.append(t[1])
                ls.append(t[2])
            else:
                raise ValueError("`tokenizer` must return `str` or `tuple` of (`str`, `int`, `int`).")
        return tokens, ss, ls

    def add_doc(self, words=None, raw=None, user_data=None, **kargs):
        '''Add a new document into the corpus and return an index of the inserted document. 
This method requires either `words` parameter or `raw` and `user_data` parameters. 
If `words` parameter is provided, `words` are expected to be already preprocessed results.
If `raw` parameter is provided, `raw` is expected to be a raw string of document which isn't preprocessed yet, and `tokenizer` will be called for preprocessing the raw document.

If you need additional parameters for a specific topic model, such as `metadata` for `tomotopy.DMRModel` or `y` for `tomotopy.SLDAModel`, you can pass it as an arbitrary keyword argument.

Parameters
----------
words : Iterable[str]
    a list of words that are already preprocessed
raw : str
    a raw string of document which isn't preprocessed yet. 
    The `raw` parameter can be used only when the `tokenizer` parameter of `__init__` is set.
user_data : Any
    an user data for `tokenizer`. The `raw` and `user_data` parameter are sent to `tokenizer`.
**kargs
    arbitrary keyword arguments for specific topic models
        '''
        if self._tokenizer:
            if not words is None: 
                raise ValueError("`raw` is required when `tokenizer` or `batch_tokenizer` is provided.")
            if not type(raw) is str:
                raise ValueError("`raw` must be `str` type.")
            if not raw: return -1
            tokens, ss, ls = self._tokenize(raw, user_data=user_data)
            self._docs.append((tokens, raw, ss, ls, kargs))
        else:
            if not raw is None: 
                raise ValueError("`words` is required when neither `tokenizer` nor `batch_tokenizer` is provided.")
            if type(words) is str:
                raise ValueError("`words` must not be `str`, but `iterable` of `str` type.")
            if not words: return -1
            self._docs.append(([self._vocab.to_id(w) for w in words], kargs))
        return len(self._docs) - 1
    
    def process(self, data_feeder):
        '''Add multiple documents into the corpus through a given iterator `data_feeder` and return the number of documents inserted.

Parameters
----------
data_feeder : Iterable[Union[str, Tuple[str, Any], Tuple[str, Any, dict]]]
    any iterable yielding a str `raw`, a tuple of (`raw`, `user_data`) or a tuple of (`raw`, `user_data`, `arbitrary_keyword_args`). 
        '''
        res = []
        num = 0
        for d in data_feeder:
            num += 1
            if type(d) is tuple and len(d) == 2:
                res.append((*d, {}))
            elif type(d) is tuple and len(d) == 3:
                res.append(d)
            elif type(d) is str:
                res.append((d, None, {}))
            else:
                raise ValueError("`data_feeder` must return an iterable of str, of Tuple[str, Any] or Tuple[str, Any, dict]")

            if len(res) >= self._batch_size:
                for raw, user_data, kargs in res:
                    self.add_doc(raw=raw, user_data=user_data, **kargs)
                res.clear()
        
        for raw, user_data, kargs in res:
            self.add_doc(raw=raw, user_data=user_data, **kargs)
        
        return num

    def save(self, filename:str):
        '''Save the current instance into the file `filename`. 

Parameters
----------
filename : str
    a path for the file where the instance is saved
        '''
        import pickle
        tok, st = self._tokenizer, self._stopwords
        self._tokenizer = self._tokenizer and True
        self._stopwords = None
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        self._tokenizer = tok
        self._stopwords = st

    @staticmethod
    def load(filename:str):
        '''Load and return an instance from the file `filename`

Parameters
----------
filename : str
    a path for the file to be loaded
        '''
        import pickle
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        obj._stopwords = lambda x : False
        return obj

    def __len__(self):
        return len(self._docs)

class SimpleTokenizer:
    '''`SimpleTokenizer` provided a simple word-tokenizing utility with an arbitrary stemmer.'''
    def __init__(self, stemmer=None, pattern:str=None):
        '''Parameters
----------
stemmer : Callable[str, str]
    a callable object for stemming words. If this value is set to `None`, words are not stemmed.
pattern : str
    a regex pattern for extracting tokens

Here is an example of using SimpleTokenizer with NLTK for stemming.

.. include:: ./auto_labeling_code_with_porter.rst
'''
        import re
        self._pat = re.compile(pattern or r"""[^\s.,;:'"?!<>(){}\[\]\\/`~@#$%^&*|]+""")
        if stemmer and not callable(stemmer):
            raise ValueError("`stemmer` must be callable.")
        self._stemmer = stemmer or None

    def __call__(self, raw:str, user_data=None):
        if self._stemmer:
            for g in self._pat.finditer(raw.lower()):
                start, end = g.span(0)
                yield self._stemmer(g.group(0)), start, end - start
        else:
            for g in self._pat.finditer(raw.lower()):
                start, end = g.span(0)
                yield g.group(0), start, end - start

import os
if os.environ.get('TOMOTOPY_LANG') == 'kr':
    __doc__ = """`tomotopy.utils` 서브모듈은 토픽 모델링에 유용한 여러 유틸리티를 제공합니다. 
`tomotopy.utils.Corpus` 클래스는 대량의 문헌을 관리할 수 있게 돕습니다. `Corpus`에 입력된 문헌들은 다양한 토픽 모델에 바로 입력될 수 있습니다.
또한 코퍼스 전처리 결과를 파일에 저장함으로써 필요에 따라 다시 코퍼스를 파일에서 읽어들여 원하는 토픽 모델에 입력할 수 있습니다.
    """
    __pdoc__ = {}
    __pdoc__['Corpus'] = """`Corpus`는 대량의 문헌을 간편하게 다룰 수 있게 도와주는 유틸리티 클래스입니다.
    `Corpus` 클래스의 인스턴스는 여러 개의 문헌을 포함할 수 있으며, 토픽 모델 클래스에 파라미터로 직접 넘겨질 수 있습니다.

Parameters
----------
tokenizer : Callable[[str, Any], Iterable[Union[str, Tuple[str, int, int]]]]
    비정제 문헌을 처리하는 데에 사용되는 호출 가능한 객체. `tokenizer`가 None이 아닌 값으로 주어진 경우, `tomotopy.utils.Corpus.add_doc` 메소드를 호출할 때 `raw` 및 `user_data` 파라미터를 사용할 수 있습니다.
    `tokenizer`는 인수로 `raw`와 `user_data` 2개를 받으며, 처리 결과로 `str`(정제된 단어) 혹은 Tuple[`str`, `int`, `int`] (정제된 단어, 단어 시작 위치, 단어 길이)의 iterable을 반환해야 합니다.
batch_size : int
    `tomotopy.utils.Corpus.process` 메소드는 대량의 문헌을 읽어들인 후 `tomotopy.utils.Corpus.add_doc`으로 넘깁니다. 이 때 한번에 읽어들이는 문헌의 개수를 `batch_size`로 지정할 수 있습니다.
stopwords : Iterable[str]
    `tomotopy.utils.Corpus.add_doc`가 호출될 때, `stopwords`에 포함된 단어들은 처리 단계에서 등록되지 않고 제외됩니다.
    `stopwords`가 호출가능한 경우, `stopwords(word) == True`이면 word는 불용어 처리되어 제외됩니다."""

    __pdoc__['Corpus.add_doc'] = """새 문헌을 코퍼스에 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.
이 메소드는 `words` 파라미터나 `raw`, `user_data` 파라미터 둘 중 하나를 요구합니다.
`words` 파라미터를 사용할 경우, `words`는 이미 전처리된 단어들의 리스트여야 합니다.
`raw` 파라미터를 사용할 경우, `raw`는 정제되기 전 문헌의 str이며, `tokenizer`가 이 비정제문헌을 처리하기 위해 호출됩니다.

만약 `tomotopy.DMRModel`의 `metadata`나 `tomotopy.SLDAModel`의 `y`처럼 특정한 토픽 모델에 필요한 추가 파라미터가 있다면 임의 키워드 인자로 넘겨줄 수 있습니다.

Parameters
----------
words : Iterable[str]
    이미 전처리된 단어들의 리스트
raw : str
    전처리되기 이전의 문헌.
    이 파라미터를 사용하려면 인스턴스 생성시 `tokenizer` 파라미터를 넣어줘야 합니다.
user_data : Any
    `tokenizer`에 넘어가는 유저 데이터.  `raw`와 `user_data` 파라미터가 함께 `tokenizer`로 넘어갑니다.
**kargs
    추가적인 파라미터를 위한 임의 키워드 인자"""
    __pdoc__['Corpus.process'] = """이터레이터 `data_feeder`를 통해 다수의 문헌을 코퍼스에 추가하고, 추가된 문헌의 개수를 반환합니다.

Parameters
----------
data_feeder : Iterable[Union[str, Tuple[str, Any], Tuple[str, Any, dict]]]
    문자열 `raw`이나, 튜플 (`raw`, `user_data`), 혹은 튜플 (`raw`, `user_data`, `kargs`) 를 반환하는 이터레이터. """
    __pdoc__['Corpus.save'] = """현재 인스턴스를 파일 `filename`에 저장합니다.. 

Parameters
----------
filename : str
    인스턴스가 저장될 파일의 경로"""
    __pdoc__['Corpus.load'] = """파일 `filename`로부터 인스턴스를 읽어들여 반환합니다.

Parameters
----------
filename : str
    읽어들일 파일의 경로"""
    
    __pdoc__['SimpleTokenizer'] = """`SimpleTokenizer`는 임의의 스테머를 사용할 수 있는 단순한 단어 분리 유틸리티입니다.

Parameters
----------
stemmer : Callable[str, str]
    단어를 스테밍하는데 사용되는 호출가능한 객체. 만약 이 값이 `None`이라면 스테밍은 사용되지 않습니다.
pattern : str
    토큰을 추출하는데 사용할 정규식 패턴

SimpleTokenizer와 NLTK를 사용하여 스테밍을 하는 예제는 다음과 같습니다.

.. include:: ./auto_labeling_code_with_porter.rst"""

del os
