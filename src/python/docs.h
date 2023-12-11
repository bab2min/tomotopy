#pragma once
#include <Python.h>

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN(name, en) PyDoc_STRVAR(name, en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#define DOC_VARIABLE_EN_KO(name, en, ko) PyDoc_STRVAR(name, en)
#endif

/*
    class Document
*/
DOC_SIGNATURE_EN_KO(Document___init____doc__,
    "Document()",
    u8R""(This type provides abstract model to access documents to be used Topic Model.

An instance of this type can be acquired from `tomotopy.LDAModel.make_doc` method or `tomotopy.LDAModel.docs` member of each Topic Model instance.)"",
u8R""(이 타입은 토픽 모델에 사용되는 문헌들에 접근할 수 있는 추상 인터페이스을 제공합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_topics__doc__,
    "get_topics(self, top_n=10, from_pseudo_doc=False)",
    u8R""(Return the `top_n` topics with its probability of the document.

Parameters
----------
top_n : int
    the `n` in "top-n"
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    If True, it returns the topic distribution of its pseudo document. Only valid for `tomotopy.PTModel`.
)"",
u8R""(현재 문헌의 상위 `top_n`개의 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다.
    
Parameters
----------
top_n : int
    "상위-n"에서 n의 값
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    참일 경우 가상 문헌의 토픽 분포를 반환합니다. `tomotopy.PTModel`에서만 유효합니다.
)"");

DOC_SIGNATURE_EN_KO(Document_get_topic_dist__doc__,
    "get_topic_dist(self, normalize=True, from_pseudo_doc=False)",
    u8R""(Return a distribution of the topics in the document.

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    If True, it returns the topic distribution of its pseudo document. Only valid for `tomotopy.PTModel`.
)"",
u8R""(현재 문헌의 토픽 확률 분포를 `list` 형태로 반환합니다.

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    참일 경우 가상 문헌의 토픽 분포를 반환합니다. `tomotopy.PTModel`에서만 유효합니다.
)"");

DOC_SIGNATURE_EN_KO(Document_get_sub_topics__doc__,
    "get_sub_topics(self, top_n=10)",
    u8R""(.. versionadded:: 0.5.0

Return the `top_n` sub topics with its probability of the document. (for only `tomotopy.PAModel`))"",
u8R""(.. versionadded:: 0.5.0

현재 문헌의 상위 `top_n`개의 하위 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다. (`tomotopy.PAModel` 전용))"");

DOC_SIGNATURE_EN_KO(Document_get_sub_topic_dist__doc__,
    "get_sub_topic_dist(self, normalize=True)",
    u8R""(.. versionadded:: 0.5.0

Return a distribution of the sub topics in the document. (for only `tomotopy.PAModel`)

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.)"",
    u8R""(.. versionadded:: 0.5.0

현재 문헌의 하위 토픽 확률 분포를 `list` 형태로 반환합니다. (`tomotopy.PAModel` 전용)

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_SIGNATURE_EN_KO(Document_get_words__doc__,
    "get_words(self, top_n=10)",
    u8R""(.. versionadded:: 0.4.2

Return the `top_n` words with its probability of the document.)"",
u8R""(.. versionadded:: 0.4.2

현재 문헌의 상위 `top_n`개의 단어와 그 확률을 `tuple`의 `list` 형태로 반환합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_count_vector__doc__,
    "get_count_vector(self)",
    u8R""(.. versionadded:: 0.7.0

Return a count vector for the current document.)"",
u8R""(.. versionadded:: 0.7.0

현재 문헌의 카운트 벡터를 반환합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_ll__doc__,
    "get_ll(self)",
    u8R""(.. versionadded:: 0.10.0

Return total log-likelihood for the current document.)"",
u8R""(.. versionadded:: 0.10.0

현재 문헌의 로그가능도 총합을 반환합니다.)"");

DOC_VARIABLE_EN_KO(Document_words__doc__,
    u8R""(a `list` of IDs for each word (read-only))"",
    u8R""(문헌 내 단어들의 ID가 담긴 `list` (읽기전용))"");

DOC_VARIABLE_EN_KO(Document_weight__doc__,
    u8R""(a weight of the document (read-only))"",
    u8R""(문헌의 가중치 (읽기전용))"");

DOC_VARIABLE_EN_KO(Document_topics__doc__,
    u8R""(a `list` of topics for each word (read-only)

This represents super topics in `tomotopy.PAModel` and `tomotopy.HPAModel` model.)"",
u8R""(문헌의 단어들이 각각 할당된 토픽을 보여주는 `list` (읽기 전용)

`tomotopy.PAModel`와 `tomotopy.HPAModel` 모형에서는 이 값이 상위토픽의 ID를 가리킵니다.)"");

DOC_VARIABLE_EN_KO(Document_uid__doc__,
    u8R""(a unique id of the document (read-only))"",
    u8R""(문헌의 고유 ID (읽기전용))"");

DOC_VARIABLE_EN_KO(Document_metadata__doc__,
    u8R""(categorical metadata of the document (for only `tomotopy.DMRModel` and `tomotopy.GDMRModel` model, read-only))"",
    u8R""(문헌의 범주형 메타데이터 (`tomotopy.DMRModel`과 `tomotopy.GDMRModel` 모형에서만 사용됨, 읽기전용))"");

DOC_VARIABLE_EN_KO(Document_multi_metadata__doc__,
    u8R""(categorical multiple metadata of the document (for only `tomotopy.DMRModel` and `tomotopy.GDMRModel` model, read-only)

.. versionadded:: 0.12.0)"",
u8R""(문헌의 범주형 메타데이터 (`tomotopy.DMRModel`과 `tomotopy.GDMRModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.12.0)"");

DOC_VARIABLE_EN_KO(Document_numeric_metadata__doc__,
    u8R""(continuous numeric metadata of the document (for only `tomotopy.GDMRModel` model, read-only)

.. versionadded:: 0.11.0)"",
u8R""(문헌의 연속형 숫자 메타데이터 (`tomotopy.GDMRModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.11.0)"");

DOC_VARIABLE_EN_KO(Document_subtopics__doc__,
    u8R""(a `list` of sub topics for each word (for only `tomotopy.PAModel` and `tomotopy.HPAModel` model, read-only))"",
    u8R""(문헌의 단어들이 각각 할당된 하위 토픽을 보여주는 `list` (`tomotopy.PAModel`와 `tomotopy.HPAModel` 모형에서만 사용됨, 읽기전용))"");

DOC_VARIABLE_EN_KO(Document_windows__doc__,
    u8R""(a `list` of window IDs for each word (for only `tomotopy.MGLDAModel` model, read-only))"",
    u8R""(문헌의 단어들이 할당된 윈도우의 ID를 보여주는 `list` (`tomotopy.MGLDAModel` 모형에서만 사용됨, 읽기전용))"");

DOC_VARIABLE_EN_KO(Document_path__doc__,
    u8R""(a `list` of topic ids by depth for a given document (for only `tomotopy.HLDAModel` model, read-only)

.. versionadded:: 0.7.1)"",
u8R""(주어진 문헌에 대한 깊이별 토픽 번호의 `list` (`tomotopy.HLDAModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.1)"");

DOC_VARIABLE_EN_KO(Document_beta__doc__,
    u8R""(a `list` of beta parameters for each topic (for only `tomotopy.CTModel` model, read-only)

.. versionadded:: 0.2.0)"",
u8R""(문헌의 각 토픽별 beta 파라미터를 보여주는 `list` (`tomotopy.CTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.2.0)"");

DOC_VARIABLE_EN_KO(Document_vars__doc__,
    u8R""(a `list` of response variables (for only `tomotopy.SLDAModel` model, read-only)

.. versionadded:: 0.2.0)"",
u8R""(문헌의 응답 변수를 보여주는 `list` (`tomotopy.SLDAModel` 모형에서만 사용됨 , 읽기전용)

.. versionadded:: 0.2.0)"");

DOC_VARIABLE_EN_KO(Document_labels__doc__,
    u8R""(a `list` of (label, list of probabilties of each topic belonging to the label) of the document (for only `tomotopy.LLDAModel` and `tomotopy.PLDAModel` models, read-only)

.. versionadded:: 0.3.0)"",
u8R""(문헌에 매겨진 (레이블, 레이블에 속하는 각 주제의 확률들)의 `list` (`tomotopy.LLDAModel`, `tomotopy.PLDAModel` 모형에서만 사용됨 , 읽기전용)

.. versionadded:: 0.3.0)"");

DOC_VARIABLE_EN_KO(Document_eta__doc__,
    u8R""(a `list` of eta parameters(topic distribution) for the current document (for only `tomotopy.DTModel` model, read-only)

.. versionadded:: 0.7.0)"",
u8R""(문헌의 eta 파라미터(토픽 분포)를 나타내는 `list` (`tomotopy.DTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.0)"");

DOC_VARIABLE_EN_KO(Document_timepoint__doc__,
    u8R""(a timepoint of the document (for only `tomotopy.DTModel` model, read-only)

.. versionadded:: 0.7.0)"",
u8R""(문헌의 시점 (`tomotopy.DTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.0)"");

DOC_VARIABLE_EN_KO(Document_raw__doc__,
    u8R""(a raw text of the document (read-only))"",
    u8R""(문헌의 가공되지 않는 전체 텍스트 (읽기전용))"");

DOC_VARIABLE_EN_KO(Document_span__doc__,
    u8R""(a span (tuple of a start position and a end position in bytes) for each word token in the document (read-only))"",
    u8R""(문헌의 각 단어 토큰의 구간(바이트 단위 시작 지점과 끝 지점의 tuple) (읽기전용))"");

DOC_VARIABLE_EN_KO(Document_pseudo_doc_id__doc__,
    u8R""(id of a pseudo document where the document is allocated to (for only `tomotopy.PTModel` model, read-only)

.. versionadded:: 0.11.0)"",
u8R""(문헌이 할당된 가상 문헌의 id (`tomotopy.PTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.11.0)"");

/*
    class LDA
*/
DOC_SIGNATURE_EN_KO(LDA___init____doc__,
    "LDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Latent Dirichlet Allocation(LDA) topic model and its implementation is based on following papers:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Latent Dirichlet Allocation(LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0    
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 범위의 정수.
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(LDA_add_doc__doc__,
    "add_doc(self, words, ignore_empty_words=True)",
    u8R""(Add a new document into the model instance and return an index of the inserted document. This method should be called before calling the `tomotopy.LDAModel.train`.

.. versionchanged:: 0.12.3

    A new argument `ignore_empty_words` was added.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다. 이 메소드는 `tomotopy.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable)"");

DOC_SIGNATURE_EN_KO(LDA_add_corpus__doc__,
    "add_corpus(self, corpus, transform=None)",
    u8R""(.. versionadded:: 0.10.0

Add new documents into the model instance using `tomotopy.utils.Corpus` and return an instance of corpus that contains the inserted documents. 
This method should be called before calling the `tomotopy.LDAModel.train`.

Parameters
----------
corpus : tomotopy.utils.Corpus
    corpus that contains documents to be added
transform : Callable[dict, dict]
    a callable object to manipulate arbitrary keyword arguments for a specific topic model
)"",
u8R""(.. versionadded:: 0.10.0

코퍼스를 이용해 현재 모델에 새로운 문헌들을 추가하고 추가된 문헌로 구성된 새 코퍼스를 반환합니다. 
이 메소드는 `tomotopy.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.
Parameters
----------
corpus : tomotopy.utils.Corpus
    토픽 모델에 추가될 문헌들로 구성된 코퍼스
transform : Callable[dict, dict]
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(LDA_make_doc__doc__,
    "make_doc(self, words)",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다..

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
)"");

DOC_SIGNATURE_EN_KO(LDA_set_word_prior__doc__,
    "set_word_prior(self, word, prior)",
    u8R""(.. versionadded:: 0.6.0

Set word-topic prior. This method should be called before calling the `tomotopy.LDAModel.train`.

Parameters
----------
word : str
    a word to be set
prior : Union[Iterable[float], Dict[int, float]]
	topic distribution of `word` whose length is equal to `tomotopy.LDAModel.k`

Note
----
Since version 0.12.6, this method can accept a dictionary type parameter as well as a list type parameter for `prior`.
The key of the dictionary is the topic id and the value is the prior of the topic. If the prior of a topic is not set, the default value is set to `eta` parameter of the model.
```python
>>> model = tp.LDAModel(k=3, eta=0.01)
>>> model.set_word_prior('apple', [0.01, 0.9, 0.01])
>>> model.set_word_prior('apple', {1: 0.9}) # same effect as above
```
)"",
u8R""(.. versionadded:: 0.6.0

어휘-주제 사전 분포를 설정합니다. 이 메소드는 `tomotopy.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.

Parameters
----------
word : str
    설정할 어휘
prior : Union[Iterable[float], Dict[int, float]]
    어휘 `word`의 주제 분포. `prior`의 길이는 `tomotopy.LDAModel.k`와 동일해야 합니다.

Note
----
0.12.6 버전부터 이 메소드는 `prior`에 리스트 타입 파라미터 외에도 딕셔너리 타입 파라미터를 받을 수 있습니다.
딕셔너리의 키는 주제의 id이며 값은 사전 주제 분포입니다. 만약 주제의 사전 분포가 설정되지 않았을 경우, 기본값으로 모델의 `eta` 파라미터로 설정됩니다.
```python
>>> model = tp.LDAModel(k=3, eta=0.01)
>>> model.set_word_prior('apple', [0.01, 0.9, 0.01])
>>> model.set_word_prior('apple', {1: 0.9}) # 위와 동일한 효과
```
)"");

DOC_SIGNATURE_EN_KO(LDA_get_word_prior__doc__,
    "get_word_prior(self, word)",
    u8R""(.. versionadded:: 0.6.0

Return word-topic prior for `word`. If there is no set prior for `word`, an empty list is returned.

Parameters
----------
word : str
    a word
)"",
u8R""(.. versionadded:: 0.6.0

`word`에 대한 사전 주제 분포를 반환합니다. 별도로 설정된 값이 없을 경우 빈 리스트가 반환됩니다.

Parameters
----------
word : str
    어휘
)"");

DOC_SIGNATURE_EN_KO(LDA_train__doc__,
    "train(self, iter=10, workers=0, parallel=0, freeze_topics=False, callback_interval=10, callback=None, show_progress=False)",
    u8R""(Train the model using Gibbs-sampling with `iter` iterations. Return `None`. 
After calling this method, you cannot `tomotopy.LDAModel.add_doc` or `tomotopy.LDAModel.set_word_prior` more.

Parameters
----------
iter : int
    the number of iterations of Gibbs-sampling
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for training. the default value is `tomotopy.ParallelScheme.DEFAULT` which means that tomotopy selects the best scheme by model.
freeze_topics : bool
    .. versionadded:: 0.10.1

    prevents to create a new topic when training. Only valid for `tomotopy.HLDAModel`
callback_interval : int
    .. versionadded:: 0.12.6

    the interval of calling `callback` function. If `callback_interval` <= 0, `callback` function is called at the beginning and the end of training.
callback : Callable[[tomotopy.LDAModel, int, int], None]
    .. versionadded:: 0.12.6

    a callable object which is called every `callback_interval` iterations. 
    It receives three arguments: the current model, the current number of iterations, and the total number of iterations.
show_progress : bool
    .. versionadded:: 0.12.6

    If `True`, it shows progress bar during training using `tqdm` package.
)"",
u8R""(깁스 샘플링을 `iter` 회 반복하여 현재 모델을 학습시킵니다. 반환값은 `None`입니다. 
이 메소드가 호출된 이후에는 더 이상 `tomotopy.LDAModel.add_doc`로 현재 모델에 새로운 학습 문헌을 추가시킬 수 없습니다.

Parameters
----------
iter : int
    깁스 샘플링의 반복 횟수
workers : int
    깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
    만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0

    학습에 사용할 병렬화 방법. 기본값은 ParallelScheme.DEFAULT로 이는 모델에 따라 최적의 방법을 tomotopy가 알아서 선택하도록 합니다.
freeze_topics : bool
    .. versionadded:: 0.10.1

    학습 시 새로운 토픽을 생성하지 못하도록 합니다. 이 파라미터는 오직 `tomotopy.HLDAModel`에만 유효합니다.
callback_interval : int
    .. versionadded:: 0.12.6

    `callback` 함수를 호출하는 간격. `callback_interval` <= 0일 경우 학습 시작과 종료 시에만 `callback` 함수가 호출됩니다.
callback : Callable[[tomotopy.LDAModel, int, int], None]
    .. versionadded:: 0.12.6

    학습 과정에서 `callback_interval` 마다 호출되는 호출가능한 객체. 
    이 함수는 세 개의 인자를 받습니다: 현재 모델, 현재까지의 반복 횟수, 총 반복 횟수.
show_progress : bool
    .. versionadded:: 0.12.6

    `True`일 경우 `tqdm` 패키지를 이용해 학습 진행 상황을 표시합니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_get_topic_words__doc__,
    "get_topic_words(self, topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    an integer in range [0, `k`), indicating the topic
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(LDA_get_topic_word_dist__doc__,
    "get_topic_word_dist(self, topic_id, normalize=True)",
    u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_get_count_by_topics__doc__,
    "get_count_by_topics(self)",
    u8R""(Return the number of words allocated to each topic.)"",
    u8R""(각각의 토픽에 할당된 단어의 개수를 `list`형태로 반환합니다.)"");

DOC_SIGNATURE_EN_KO(LDA_infer__doc__,
    "infer(self, doc, iter=100, tolerance=-1, workers=0, parallel=0, together=False, transform=None)",
    u8R""(Return the inferred topic distribution from unseen `doc`s.

Parameters
----------
doc : Union[tomotopy.Document, Iterable[tomotopy.Document], tomotopy.utils.Corpus]
    an instance of `tomotopy.Document` or a `list` of instances of `tomotopy.Document` to be inferred by the model.
    It can be acquired from `tomotopy.LDAModel.make_doc` method.

    .. versionchanged:: 0.10.0

        Since version 0.10.0, `infer` can receive a raw corpus instance of `tomotopy.utils.Corpus`. 
        In this case, you don't need to call `make_doc`. `infer` would generate documents bound to the model, estimate its topic distributions and
        return a corpus contains generated documents as the result.
iter : int
    an integer indicating the number of iteration to estimate the distribution of topics of `doc`.
    The higher value will generate a more accuracy result.
tolerance : float
    isn't currently used.
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for inference. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.
together : bool
    all `doc`s are infered together in one process if True, otherwise each `doc` is infered independently. Its default value is `False`.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    a callable object to manipulate arbitrary keyword arguments for a specific topic model. 
    Available when `doc` is given as an instance of `tomotopy.utils.Corpus`.

Returns
-------
result : Union[List[float], List[List[float]], tomotopy.utils.Corpus]
    If `doc` is given as a single `tomotopy.Document`, `result` is a single `List[float]` indicating its topic distribution.
    
    If `doc` is given as a list of `tomotopy.Document`s, `result` is a list of `List[float]` indicating topic distributions for each document.
    
    If `doc` is given as an instance of `tomotopy.utils.Corpus`, `result` is another instance of `tomotopy.utils.Corpus` which contains infered documents.
    You can get topic distribution for each document using `tomotopy.Document.get_topic_dist`.
log_ll : List[float]
    a list of log-likelihoods for each `doc`s
)"",
u8R""(새로운 문헌인 `doc`에 대해 각각의 주제 분포를 추론하여 반환합니다.
반환 타입은 (`doc`의 주제 분포, 로그가능도) 또는 (`doc`의 주제 분포로 구성된 `list`, 로그가능도)입니다.

Parameters
----------
doc : Union[tomotopy.Document, Iterable[tomotopy.Document], tomotopy.utils.Corpus]
    추론에 사용할 `tomotopy.Document`의 인스턴스이거나 이 인스턴스들의 `list`.
    이 인스턴스들은 `tomotopy.LDAModel.make_doc` 메소드를 통해 얻을 수 있습니다.

    .. versionchanged:: 0.10.0

        0.10.0버전부터 `infer`는 `tomotopy.utils.Corpus`의 인스턴스를 직접 입력 받을 수 있습니다. 
        이 경우 `make_doc`를 사용할 필요 없이 `infer`가 직접 모델에 맞춰진 문헌을 생성하고 이를 이용해 토픽 분포를 추정하며,
        결과로 생성된 문헌들이 포함된 `tomotopy.utils.Corpus`를 반환합니다.
iter : int
    `doc`의 주제 분포를 추론하기 위해 학습을 반복할 횟수입니다.
    이 값이 클 수록 더 정확한 결과를 낼 수 있습니다.
tolerance : float
    현재는 사용되지 않음
workers : int
    깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
    만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0

    추론에 사용할 병렬화 방법. 기본값은 ParallelScheme.DEFAULT로 이는 모델에 따라 최적의 방법을 tomotopy가 알아서 선택하도록 합니다.
together : bool
    이 값이 True인 경우 입력한 `doc` 문헌들을 한 번에 모델에 넣고 추론을 진행합니다.
    False인 경우 각각의 문헌들을 별도로 모델에 넣어 추론합니다. 기본값은 `False`입니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체.
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우에만 사용 가능합니다.

Returns
-------
result : Union[List[float], List[List[float]], tomotopy.utils.Corpus]
    `doc`이 `tomotopy.Document`로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`입니다.
    
    `doc`이 `tomotopy.Document`의 list로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`의 list입니다.
    
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우, `result`는 추론된 결과 문서들을 담고 있는, `tomotopy.utils.Corpus`의 새로운 인스턴스입니다.
    각 문헌별 토픽 분포를 얻기 위해서는 `tomotopy.Document.get_topic_dist`를 사용하면 됩니다.
log_ll : float
    각 문헌별 로그 가능도의 리스트
)"");

DOC_SIGNATURE_EN_KO(LDA_save__doc__,
    "save(self, filename, full=True)",
    u8R""(Save the model instance to file `filename`. Return `None`.

If `full` is `True`, the model with its all documents and state will be saved. If you want to train more after, use full model.
If `False`, only topic parameters of the model will be saved. This model can be only used for inference of an unseen document.

.. versionadded:: 0.6.0

Since version 0.6.0, the model file format has been changed. 
Thus model files saved in version 0.6.0 or later are not compatible with versions prior to 0.5.2.
)"",
u8R""(현재 모델을 `filename` 경로의 파일에 저장합니다. `None`을 반환합니다.

`full`이 `True`일 경우, 모델의 전체 상태가 파일에 모두 저장됩니다. 저장된 모델을 다시 읽어들여 학습(`train`)을 더 진행하고자 한다면 `full` = `True`로 하여 저장하십시오.
반면 `False`일 경우, 토픽 추론에 관련된 파라미터만 파일에 저장됩니다. 이 경우 파일의 용량은 작아지지만, 추가 학습은 불가하고 새로운 문헌에 대해 추론(`infer`)하는 것만 가능합니다.

.. versionadded:: 0.6.0

0.6.0 버전부터 모델 파일 포맷이 변경되었습니다.
따라서 0.6.0 이후 버전에서 저장된 모델 파일 포맷은 0.5.2 버전 이전과는 호환되지 않습니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_saves__doc__,
    "saves(self, full=True)",
    u8R""(.. versionadded:: 0.11.0

Serialize the model instance into `bytes` object and return it. The arguments work the same as `tomotopy.LDAModel.save`.)"",
u8R""(.. versionadded:: 0.11.0

현재 모델을 직렬화하여 `bytes`로 만든 뒤 이를 반환합니다. 인자는 `tomotopy.LDAModel.save`와 동일하게 작동합니다.)"");


DOC_SIGNATURE_EN_KO(LDA_load__doc__,
    "load(filename)",
    u8R""(Return the model instance loaded from file `filename`.)"",
    u8R""(`filename` 경로의 파일로부터 모델 인스턴스를 읽어들여 반환합니다.)"");

DOC_SIGNATURE_EN_KO(LDA_loads__doc__,
    "loads(data)",
    u8R""(Return the model instance loaded from `data` in bytes-like object.)"",
    u8R""(bytes-like object인 `data`로로부터 모델 인스턴스를 읽어들여 반환합니다.)"");

DOC_SIGNATURE_EN_KO(LDA_copy__doc__,
    "copy(self)",
    u8R""(.. versionadded:: 0.12.0

Return a new deep-copied instance of the current instance)"",
u8R""(.. versionadded:: 0.12.0

깊게 복사된 새 인스턴스를 반환합니다.)"");


DOC_SIGNATURE_EN_KO(LDA_summary__doc__,
    "summary(self, initial_hp=True, params=True, topic_word_top_n=5, file=None, flush=False)",
    u8R""(.. versionadded:: 0.9.0

print human-readable description of the current model

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
)"",
u8R""(.. versionadded:: 0.9.0

현재 모델의 요약 정보를 읽기 편한 형태로 출력합니다.

Parameters
----------
initial_hp : bool
    모델 생성 시 초기 파라미터의 표시 여부
params : bool
    현재 모델 파라미터의 표시 여부
topic_word_top_n : int
    토픽별 출력할 단어의 개수
file
    요약 정보를 출력할 대상, 기본값은 `sys.stdout`
flush : bool
    출력 스트림의 강제 flush 여부
)"");


DOC_VARIABLE_EN_KO(LDA_tw__doc__,
    u8R""(the term weighting scheme (read-only))"",
    u8R""(현재 모델의 용어 가중치 계획 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_perplexity__doc__,
    u8R""(a perplexity of the model (read-only))"",
    u8R""(현재 모델의 Perplexity (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_ll_per_word__doc__,
    u8R""(a log likelihood per-word of the model (read-only))"",
    u8R""(현재 모델의 단어당 로그 가능도 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_k__doc__,
    u8R""(K, the number of topics (read-only))"",
    u8R""(토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_alpha__doc__,
    u8R""(Dirichlet prior on the per-document topic distributions (read-only))"",
    u8R""(문헌의 토픽 분포에 대한 디리클레 분포 파라미터 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_eta__doc__,
    u8R""(the hyperparameter eta (read-only))"",
    u8R""(하이퍼 파라미터 eta (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_docs__doc__,
    u8R""(a `list`-like interface of `tomotopy.Document` in the model instance (read-only))"",
    u8R""(현재 모델에 포함된 `tomotopy.Document`에 접근할 수 있는 `list`형 인터페이스 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_vocabs__doc__,
    u8R""(a dictionary, which contains both vocabularies filtered by frequency and vocabularies actually used in modeling, as the type `tomotopy.Dictionary` (read-only))"",
    u8R""(빈도수로 필터링된 어휘와 모델에 포함된 어휘 전체를 포함하는 `tomotopy.Dictionary` 타입의 어휘 사전 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_num_vocabs__doc__,
    u8R""(the number of vocabuluaries after words with a smaller frequency were removed (read-only)

This value is 0 before `train` called.

.. deprecated:: 0.8.0

    Due to the confusion of its name, this property will be removed. Please use `len(used_vocabs)` instead.)"",
    u8R""(작은 빈도의 단어들을 제거한 뒤 남은 어휘의 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.

.. deprecated:: 0.8.0

    이 프로퍼티의 이름은 혼동을 일으킬 여지가 있어 제거될 예정입니다. 대신 `len(used_vocabs)`을 사용하십시오.)"");

DOC_VARIABLE_EN_KO(LDA_used_vocabs__doc__,
    u8R""(a dictionary, which contains only the vocabularies actually used in modeling, as the type `tomotopy.Dictionary` (read-only)

.. versionadded:: 0.8.0)"",
u8R""(모델에 실제로 사용된 어휘만을 포함하는 `tomotopy.Dictionary` 타입의 어휘 사전 (읽기전용)

.. versionadded:: 0.8.0)"");

DOC_VARIABLE_EN_KO(LDA_vocab_freq__doc__,
    u8R""(a `list` of vocabulary frequencies which contains both vocabularies filtered by frequency and vocabularies actually used in modeling (read-only))"",
    u8R""(빈도수로 필터링된 어휘와 현재 모델에 포함된 어휘 전체의 빈도를 보여주는 `list` (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_used_vocab_freq__doc__,
    u8R""(a `list` of vocabulary frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0)"",
u8R""(모델에 실제로 사용된 어휘들의 빈도를 보여주는 `list` (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_used_vocab_weighted_freq__doc__,
    u8R""(a `list` of term-weighted vocabulary frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.12.1)"",
u8R""(모델에 실제로 사용된 어휘들의 빈도(용어 가중치 적용됨)를 보여주는 `list` (읽기전용))"");


DOC_VARIABLE_EN_KO(LDA_vocab_df__doc__,
    u8R""(a `list` of vocabulary document-frequencies which contains both vocabularies filtered by frequency and vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0)"",
u8R""(빈도수로 필터링된 어휘와 현재 모델에 포함된 어휘 전체의 문헌빈도를 보여주는 `list` (읽기전용)

.. versionadded:: 0.8.0)"");

DOC_VARIABLE_EN_KO(LDA_used_vocab_df__doc__,
    u8R""(a `list` of vocabulary document-frequencies which contains only vocabularies actually used in modeling (read-only)

.. versionadded:: 0.8.0)"",
u8R""(모델에 실제로 사용된 어휘들의 문헌빈도를 보여주는 `list` (읽기전용)

.. versionadded:: 0.8.0)"");


DOC_VARIABLE_EN_KO(LDA_num_words__doc__,
    u8R""(the number of total words (read-only)

This value is 0 before `train` called.)"",
u8R""(현재 모델에 포함된 문헌들 전체의 단어 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.)"");

DOC_VARIABLE_EN_KO(LDA_global_step__doc__,
    u8R""(the total number of iterations of training (read-only)

.. versionadded:: 0.9.0)"",
u8R""(현재까지 수행된 학습의 총 반복 횟수 (읽기전용)

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(LDA_optim_interval__doc__,
    u8R""(get or set the interval for optimizing parameters

Its default value is 10. If it is set to 0, the parameter optimization is turned off.)"",
u8R""(파라미터 최적화의 주기를 얻거나 설정합니다.

기본값은 10이며, 0으로 설정할 경우 학습 과정에서 파라미터 최적화를 수행하지 않습니다.)"");

DOC_VARIABLE_EN_KO(LDA_burn_in__doc__,
    u8R""(get or set the burn-in iterations for optimizing parameters

Its default value is 0.)"",
u8R""(파라미터 학습 초기의 Burn-in 단계의 반복 횟수를 얻거나 설정합니다.

기본값은 0입니다.)"");

DOC_VARIABLE_EN_KO(LDA_removed_top_words__doc__,
    u8R""(a `list` of `str` which is a word removed from the model if you set `rm_top` greater than 0 at initializing the model (read-only))"",
    u8R""(모델 생성시 `rm_top` 파라미터를 1 이상으로 설정한 경우, 빈도수가 높아서 모델에서 제외된 단어의 목록을 보여줍니다. (읽기전용))"");

/*
    class DMR
*/
DOC_SIGNATURE_EN_KO(DMR___init____doc__,
    "DMRModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=0.0000000001, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Dirichlet Multinomial Regression(DMR) topic model and its implementation is based on following papers:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Dirichlet Multinomial Regression(DMR) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0    
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 범위의 정수.
alpha : Union[float, Iterable[float]]
    `lambdas` 파라미터의 평균의 exp의 초기값, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
sigma : float
    `lambdas` 파라미터의 표준 편차
alpha_epsilon : float
    `exp(lambdas)`가 0이 되는 것을 방지하는 평탄화 계수
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(DMR_add_doc__doc__,
    "add_doc(self, words, metadata='', multi_metadata=[], ignore_empty_words=True)",
    u8R""(Add a new document into the model instance with `metadata` and return an index of the inserted document.

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
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
metadata : str
    문헌의 메타데이터 (예로 저자나 제목, 작성연도 등)
multi_metadata : Iterable[str]
    문헌의 메타데이터 (다중 값이 필요한 경우 사용하십시오)
)"");

DOC_SIGNATURE_EN_KO(DMR_make_doc__doc__,
    "make_doc(self, words, metadata='', multi_metadata=[])",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.LDAModel.infer` method.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iteratable of `str`
metadata : str
    metadata of the document (e.g., author, title or year)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
metadata : str
    문헌의 메타데이터 (예를 들어 저자나 제목, 작성연도 등)
multi_metadata : Iterable[str]
    문헌의 메타데이터 (다중 값이 필요한 경우 사용하십시오)
)"");

DOC_SIGNATURE_EN_KO(DMR_get_topic_prior__doc__,
    "get_topic_prior(self, metadata='', multi_metadata=[], raw=False)",
    u8R""(.. versionadded:: 0.12.0

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
)"",
u8R""(.. versionadded:: 0.12.0

주어진 `metadata`와 `multi_metadata`에 대해 토픽의 사전 분포를 계산합니다.
`raw`가 참인 경우 `exp()`가 적용되기 전의 값이 반환되며, 그 외에는 `exp()`가 적용된 값이 반환됩니다.

토픽의 사전분포는 다음과 같이 계산됩니다:

`np.dot(lambda_[:, id(metadata)], np.concat([[1], multi_hot(multi_metadata)]))`

여기서 `idx(metadata)`와 `multi_hot(multi_metadata)`는 각각
주어진 `metadata`의 정수 인덱스 번호와 `multi_metadata`를 multi-hot 인코딩한, 0 혹은 1로 구성된 벡터입니다.

Parameters
----------
metadata : str
    문헌의 메타데이터 (예를 들어 저자나 제목, 작성연도 등)
multi_metadata : Iterable[str]
    문헌의 메타데이터 (다중 값이 필요한 경우 사용하십시오)
raw : bool
    참일 경우 파라미터에 `exp()`가 적용되지 않은 값이 반환됩니다.
)"");

DOC_VARIABLE_EN_KO(DMR_f__doc__,
    u8R""(the number of metadata features (read-only))"",
    u8R""(메타데이터 자질 종류의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_sigma__doc__,
    u8R""(the hyperparameter sigma (read-only))"",
    u8R""(하이퍼 파라미터 sigma (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_alpha_epsilon__doc__,
    u8R""(the smooting value alpha-epsilon (read-only))"",
    u8R""(평탄화 계수 alpha-epsilon (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_metadata_dict__doc__,
    u8R""(a dictionary of metadata in type `tomotopy.Dictionary` (read-only))"",
    u8R""(`tomotopy.Dictionary` 타입의 메타데이터 사전 (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_multi_metadata_dict__doc__,
    u8R""(a dictionary of metadata in type `tomotopy.Dictionary` (read-only)

.. versionadded:: 0.12.0

    This dictionary is distinct from `metadata_dict`.
)"",
u8R""(`tomotopy.Dictionary` 타입의 메타데이터 사전 (읽기전용)

.. versionadded:: 0.12.0

    이 사전은 `metadata_dict`와는 별개입니다.
)"");

DOC_VARIABLE_EN_KO(DMR_lamdas__doc__,
    u8R""(parameter lambdas in the shape `[k, f]` (read-only)

.. warning::

    Prior to version 0.11.0, there was a bug in the lambda getter, so it yielded the wrong value. It is recommended to upgrade to version 0.11.0 or later.
)"",
u8R""(현재 모형의 lambda 파라미터을 보여주는 `[k, f]` 모양의 float array (읽기전용)

.. warning::

    0.11.0 버전 전까지는 lambda getter에 있는 버그로 잘못된 값이 출력되었습니다. 0.11.0 이후 버전으로 업그레이드하시길 권장합니다.)"");


DOC_VARIABLE_EN_KO(DMR_lamda___doc__,
    u8R""(parameter lambdas in the shape `[k, len(metadata_dict), l]` where `k` is the number of topics and `l` is the size of vector for multi_metadata (read-only)

See `tomotopy.DMRModel.get_topic_prior` for the relation between the lambda parameter and the topic prior.

.. versionadded:: 0.12.0
)"",
u8R""(현재 모형의 lambda 파라미터을 보여주는 `[k, len(metadata_dict), l]` 모양의 float array (읽기전용)

lambda 파라미터와 토픽 사전 분포 간의 관계에 대해서는 `tomotopy.DMRModel.get_topic_prior`를 참고하십시오.

.. versionadded:: 0.12.0)"");


DOC_VARIABLE_EN_KO(DMR_alpha__doc__,
    u8R""(Dirichlet prior on the per-document topic distributions for each metadata in the shape `[k, f]`. Equivalent to `np.exp(DMRModel.lambdas)` (read-only)

.. versionadded:: 0.9.0

.. warning::

    Prior to version 0.11.0, there was a bug in the lambda getter, so it yielded the wrong value. It is recommended to upgrade to version 0.11.0 or later.)"",
    u8R""(각 메타데이터별 문헌-토픽 분포의 사전 분포, `[k, f]` 모양. `np.exp(DMRModel.lambdas)`와 동일 (읽기전용)

.. versionadded:: 0.9.0

.. warning::

    0.11.0 버전 전까지는 lambda getter에 있는 버그로 잘못된 값이 출력되었습니다. 0.11.0 이후 버전으로 업그레이드하시길 권장합니다.)"");

/*
    class GDMR
*/
DOC_SIGNATURE_EN_KO(GDMR___init____doc__,
    "GDMRModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, degrees=[], alpha=0.1, eta=0.01, sigma=1.0, sigma0=3.0, decay=0, alpha_epsilon=0.0000000001, metadata_range=None, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Generalized DMR(g-DMR) topic model and its implementation is based on following papers:

> * Lee, M., & Song, M. Incorporating citation impact into analysis of research trends. Scientometrics, 1-34.

.. versionadded:: 0.8.0

.. warning::

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.DMRModel`.
    So `metadata` arguments in the older codes should be replaced with `numeric_metadata` to work in version 0.11.0.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
degrees : Iterable[int]
    a list of the degrees of Legendre polynomials for TDF(Topic Distribution Function). Its length should be equal to the number of metadata variables.

    Its default value is `[]` in which case the model doesn't use any metadata variable and as a result, it becomes the same as the LDA or DMR model. 
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
)"",
u8R""(이 타입은 Generalized DMR(g-DMR) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Lee, M., & Song, M. Incorporating citation impact into analysis of research trends. Scientometrics, 1-34.

.. versionadded:: 0.8.0

.. warning::

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
    `metadata`라는 이름으로 범주형 변수를 사용할 수 있게 변경됩니다.
    따라서 이전 코드의 `metadata` 인자를 `numeric_metadata`로 바꿔주어야 0.11.0 버전에서 작동합니다.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 범위의 정수.
degrees : Iterable[int]
    TDF(토픽 분포 함수)로 쓰일 르장드르 다항식의 차수를 나타내는 list. 길이는 메타데이터 변수의 개수와 동일해야 합니다.

    기본값은 `[]`으로 이 경우 모델은 어떤 메타데이터 변수도 포함하지 않으므로 LDA 또는 DMR 모델과 동일해집니다.
alpha : Union[float, Iterable[float]]
    `lambdas` 파라미터의 평균의 exp의 초기값, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
sigma : float
    `lambdas` 파라미터 중 비상수 항의 표준 편차
sigma0 : float
    `lambdas` 파라미터 중 상수 항의 표준 편차
decay : float
    .. versionadded:: 0.11.0

    `lambdas` 파라미터 중 고차항의 계수가 더 작아지도록하는 감쇠 지수
alpha_epsilon : float
    `exp(lambdas)`가 0이 되는 것을 방지하는 평탄화 계수
metadata_range : Iterable[Iterable[float]]
    각 메타데이터 변수의 최솟값과 최댓값을 지정하는 list. 길이는 `degrees`의 길이와 동일해야 합니다.
    
    예를 들어 `metadata_range = [(2000, 2017), (0, 1)]` 는 첫번째 변수의 범위를 2000에서 2017까지로, 두번째 변수의 범위를 0에서 1까지로 설정하겠다는 뜻입니다.
    기본값은 `None`이며, 이 경우 입력 문헌의 메타데이터로부터 최솟값과 최댓값을 찾습니다.
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(GDMR_add_doc__doc__,
    "add_doc(self, words, numeric_metadata=[], metadata='', multi_metadata=[], ignore_empty_words=True)",
    u8R""(Add a new document into the model instance with `metadata` and return an index of the inserted document.

.. versionchanged:: 0.11.0

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.DMRModel`.

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
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

.. versionchanged:: 0.11.0

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
    `metadata`라는 이름으로 범주형 변수를 사용할 수 있게 변경됩니다.

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
numeric_metadata : Iterable[float]
    문헌의 연속형 숫자 메타데이터 변수. 길이는 `degrees`의 길이와 동일해야 합니다.
metadata : str
    문헌의 범주형 메타데이터 (예를 들어 저자나 제목, 저널, 국가 등)
multi_metadata : Iterable[str]
    문헌의 메타데이터 (다중 값이 필요한 경우 사용하십시오)
)"");

DOC_SIGNATURE_EN_KO(GDMR_make_doc__doc__,
    "make_doc(self, words, numeric_metadata=[], metadata='', multi_metadata=[])",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.LDAModel.infer` method.

.. versionchanged:: 0.11.0

    Until version 0.10.2, `metadata` was used to represent numeric data and there was no argument for categorical data.
    Since version 0.11.0, the name of the previous `metadata` argument is changed to `numeric_metadata`, 
    and `metadata` is added to represent categorical data for unification with the `tomotopy.DMRModel`.

.. versionchanged:: 0.12.0

    A new argument `multi_metadata` for multiple values of metadata was added.

Parameters
----------
words : Iterable[str]
    an iteratable of `str`
numeric_metadata : Iterable[float]
    continuous numeric metadata variable of the document. Its length should be equal to the length of `degrees`.
metadata : str
    categorical metadata of the document (e.g., author, title, journal or country)
multi_metadata : Iterable[str]
    metadata of the document (for multiple values)
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

.. versionchanged:: 0.11.0

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
    `metadata`라는 이름으로 범주형 변수를 사용할 수 있게 변경됩니다.

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
numeric_metadata : Iterable[float]
    문헌의 연속형 숫자 메타데이터 변수. 길이는 `degrees`의 길이와 동일해야 합니다.
metadata : str
    문헌의 범주형 메타데이터 (예를 들어 저자나 제목, 저널, 국가 등)
multi_metadata : Iterable[str]
    문헌의 메타데이터 (다중 값이 필요한 경우 사용하십시오)
)"");

DOC_SIGNATURE_EN_KO(GDMR_tdf__doc__,
    "tdf(self, numeric_metadata, metadata='', multi_metadata=[], normalize=True)",
    u8R""(Calculate a topic distribution for given `numeric_metadata` value. It returns a list with length `k`.

.. versionchanged:: 0.11.0

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
)"",
u8R""(주어진 `metadata`에 대해 토픽 분포를 계산하여, `k` 길이의 list로 반환합니다.

.. versionchanged:: 0.11.0

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
numeric_metadata : Iterable[float]
    연속형 메타데이터 변수. 길이는 `degrees`의 길이와 동일해야 합니다.
metadata : str
    범주형 메타데이터 변수
multi_metadata : Iterable[str]
    범주형 메타데이터 변수 (여러 개를 입력해야 하는 경우 사용하십시오)
normalize : bool
    참인 경우, 각 값이 [0, 1] 범위에 있는 확률 분포를 반환합니다. 거짓인 경우 logit값을 그대로 반환합니다.
)"");


DOC_SIGNATURE_EN_KO(GDMR_tdf_linspace__doc__,
    "tdf_linspace(self, numeric_metadata_start, numeric_metadata_stop, num, metadata='', multi_metadata=[], endpoint=True, normalize=True)",
    u8R""(Calculate a topic distribution for given `metadata` value. It returns a list with length `k`.

.. versionchanged:: 0.11.0

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
)"",
u8R""(주어진 `metadata`에 대해 토픽 분포를 계산하여, `k` 길이의 list로 반환합니다.

.. versionchanged:: 0.11.0

.. versionchanged:: 0.12.0

    여러 개의 메타데이터를 입력하는데 쓰이는 `multi_metadata`가 추가되었습니다.

Parameters
----------
numeric_metadata_start : Iterable[float]
    문헌의 연속 메타데이터 변수의 시작값. 길이는 `degrees`의 길이와 동일해야 합니다.
numeric_metadata_stop : Iterable[float]
    문헌의 연속 메타데이터 변수의 끝값. 길이는 `degrees`의 길이와 동일해야 합니다.
num : Iterable[int]
    각 메타데이터 변수별로 생성할 샘플의 개수(0보다 큰 정수). 길이는 `degrees`의 길이와 동일해야 합니다.
metadata : str
    범주형 메타데이터 변수
multi_metadata : Iterable[str]
    범주형 메타데이터 변수 (여러 개를 입력해야 하는 경우 사용하십시오)
endpoint : bool
    참인 경우 `metadata_stop`이 마지막 샘플이 됩니다. 거짓인 경우 끝값이 샘플에 포함되지 않습니다. 기본값은 참입니다.
normalize : bool
    참인 경우, 각 값이 [0, 1] 범위에 있는 확률 분포를 반환합니다. 거짓인 경우 logit값을 그대로 반환합니다.
)"");


DOC_VARIABLE_EN_KO(GDMR_degrees__doc__,
    u8R""(the degrees of Legendre polynomials (read-only))"",
    u8R""(르장드르 다항식의 차수 (읽기전용))"");

DOC_VARIABLE_EN_KO(GDMR_sigma0__doc__,
    u8R""(the hyperparameter sigma0 (read-only))"",
    u8R""(하이퍼 파라미터 sigma0 (읽기전용))"");

DOC_VARIABLE_EN_KO(GDMR_decay__doc__,
    u8R""(the hyperparameter decay (read-only))"",
    u8R""(하이퍼 파라미터 decay (읽기전용))"");

DOC_VARIABLE_EN_KO(GDMR_metadata_range__doc__,
    u8R""(the ranges of each metadata variable (read-only))"",
    u8R""(각 메타데이터 변수의 범위를 나타내는 `list` (읽기전용))"");


/*
    class HDP
*/
DOC_SIGNATURE_EN_KO(HDP___init____doc__,
    "HDPModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, initial_k=2, alpha=0.1, eta=0.01, gamma=0.1, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Hierarchical Dirichlet Process(HDP) topic model and its implementation is based on following papers:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

.. versionchanged:: 0.3.0

    Since version 0.3.0, hyperparameter estimation for `alpha` and `gamma` has been added. You can turn off this estimation by setting `optim_interval` to zero.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
initial_k : int
    the initial number of topics between 2 ~ 32767
    The number of topics will be adjusted for data during training.
	
	Since version 0.3.0, the default value has been changed to 2 from 1.
alpha : float
    concentration coeficient of Dirichlet Process for document-table 
eta : float
    hyperparameter of Dirichlet distribution for topic-word
gamma : float
    concentration coeficient of Dirichlet Process for table-topic
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model
)"",
u8R""(이 타입은 Hierarchical Dirichlet Process(HDP) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

.. versionchanged:: 0.3.0

    0.3.0버전부터 `alpha`와 `gamma`에 대한 하이퍼파라미터 추정 기능이 추가되었습니다. `optim_interval`을 0으로 설정함으로써 이 기능을 끌 수 있습니다.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0    
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
initial_k : int
    초기 토픽의 개수를 지정하는 2 ~ 32767 범위의 정수.
	
	0.3.0버전부터 기본값이 1에서 2로 변경되었습니다.
alpha : float
    document-table에 대한 Dirichlet Process의 집중 계수
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
gamma : float
    table-topic에 대한 Dirichlet Process의 집중 계수
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(HDP_is_live_topic__doc__,
    "is_live_topic(self, topic_id)",
    u8R""(Return `True` if the topic `topic_id` is valid, otherwise return `False`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id`가 유효한 토픽을 가리키는 경우 `True`, 아닌 경우 `False`를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(HDP_convert_to_lda__doc__,
    "convert_to_lda(self, topic_threshold=0.0)",
    u8R""(.. versionadded:: 0.8.0

Convert the current HDP model to equivalent LDA model and return `(new_lda_model, new_topic_id)`.
Topics with proportion less than `topic_threshold` are removed in `new_lda_model`.

`new_topic_id` is an array of length `HDPModel.k` and `new_topic_id[i]` indicates a topic id of new LDA model, equivalent to topic `i` of original HDP model.
If topic `i` of original HDP model is not alive or is removed in LDA model, `new_topic_id[i]` would be `-1`.

Parameters
----------
topic_threshold : float
    Topics with proportion less than this value is removed in new LDA model.
    The default value is 0, and it means no topic except not alive is removed.
)"",
u8R""(.. versionadded:: 0.8.0

현재의 HDP 모델을 동등한 LDA모델로 변환하고, `(new_lda_mode, new_topic_id)`를 반환합니다.
이 때 `topic_threshold`보다 작은 비율의 토픽은 `new_lda_model`에서 제거됩니다.

`new_topic_id`는 길이 `HDPModel.k`의 배열이며, `new_topic_id[i]`는 새 LDA 모델에서 원 HDP 모델의 토픽 `i`와 동등한 토픽의 id를 가리킵니다.
만약 원 HDP 모델의 토픽 `i`가 유효하지 않거나, 새 LDA 모델에서 제거된 것이라면, `new_topic_id[i]`는 `-1`이 됩니다.

Parameters
----------
topic_threshold : float
    이 값보다 작은 비율의 토픽은 새 LDA 모델에서 제거됩니다.
    기본값은 0이며, 이 경우 유효하지 않는 토픽을 제외한 모든 토픽이 LDA 모델에 포함됩니다.
)"");

DOC_SIGNATURE_EN_KO(HDP_purge_dead_topics__doc__,
    "purge_dead_topics(self)",
    u8R""(.. versionadded:: 0.12.3

Purge all non-alive topics from the model and return `new_topic_ids`. After called, `HDPModel.k` shrinks to `HDPModel.live_k` and all topics of the model become live.

`new_topic_id` is an array of length `HDPModel.k` and `new_topic_id[i]` indicates a topic id of the new model, equivalent to topic `i` of previous HDP model.
If topic `i` of previous HDP model is not alive or is removed in the new model, `new_topic_id[i]` would be `-1`.
)"",
u8R""(.. versionadded:: 0.12.3

현재 모델에서 유효하지 않은 토픽을 모두 제거하고 `new_topic_ids`를 반환합니다. 호출 후에 `HDPModel.k`는 `HDPModel.live_k`값으로 줄어들며 모든 토픽은 유효한 상태가 됩니다.

`new_topic_id`는 길이 `HDPModel.k`의 배열이며, `new_topic_id[i]`는 새 모델에서 기존 HDP 모델의 토픽 `i`와 동등한 토픽의 id를 가리킵니다.
만약 기존 HDP 모델의 토픽 `i`가 유효하지 않거나, 새 모델에서 제거된 것이라면, `new_topic_id[i]`는 `-1`이 됩니다.
)"");

DOC_VARIABLE_EN_KO(HDP_gamma__doc__,
    u8R""(the hyperparameter gamma (read-only))"",
    u8R""(하이퍼 파라미터 gamma (읽기전용))"");

DOC_VARIABLE_EN_KO(HDP_live_k__doc__,
    u8R""(the number of alive topics (read-only))"",
    u8R""(현재 모델 내의 유효한 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(HDP_num_tables__doc__,
    u8R""(the number of total tables (read-only))"",
    u8R""(현재 모델 내의 총 테이블 개수 (읽기전용))"");

/*
    class MGLDA
*/
DOC_SIGNATURE_EN_KO(MGLDA___init____doc__,
    "MGLDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Multi Grain Latent Dirichlet Allocation(MG-LDA) topic model and its implementation is based on following papers:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Multi Grain Latent Dirichlet Allocation(MG-LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0    
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k_g : int
    전역 토픽의 개수를 지정하는 1 ~ 32767 사이의 정수
k_l : int
    지역 토픽의 개수를 지정하는 1 ~ 32767 사이의 정수
t : int
    문장 윈도우의 크기
alpha_g : float
    문헌-전역 토픽 디리클레 분포의 하이퍼 파라미터
alpha_l : float
    문헌-지역 토픽 디리클레 분포의 하이퍼 파라미터
alpha_mg : float
    전역/지역 선택 디리클레 분포의 하이퍼 파라미터 (전역 부분 계수)
alpha_ml : float
    전역/지역 선택 디리클레 분포의 하이퍼 파라미터 (지역 부분 계수)
eta_g : float
    전역 토픽-단어 디리클레 분포의 하이퍼 파라미터
eta_l : float
    지역 토픽-단어 디리클레 분포의 하이퍼 파라미터
gamma : float
    문장-윈도우 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(MGLDA_add_doc__doc__,
    "add_doc(self, words, delimiter='.', ignore_empty_words=True)",
    u8R""(Add a new document into the model instance and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
delimiter : str
    a sentence separator. `words` will be separated by this value into sentences.
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
delimiter : str
    문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_make_doc__doc__,
    "make_doc(self, words, delimiter='.')",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iteratable of `str`
delimiter : str
    a sentence separator. `words` will be separated by this value into sentences.
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
delimiter : str
    문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_get_topic_words__doc__,
    "get_topic_words(self, topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int 
    A number in range [0, `k_g`) indicates a global topic and 
    a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_get_topic_word_dist__doc__,
    "get_topic_word_dist(self, topic_id, normalize=True)",
    u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

Parameters
----------
topic_id : int 
    A number in range [0, `k_g`) indicates a global topic and 
    a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_VARIABLE_EN_KO(MGLDA_k_g__doc__,
    u8R""(the hyperparameter k_g (read-only))"",
    u8R""(하이퍼 파라미터 k_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_k_l__doc__,
    u8R""(the hyperparameter k_l (read-only))"",
    u8R""(하이퍼 파라미터 k_l (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_gamma__doc__,
    u8R""(the hyperparameter gamma (read-only))"",
    u8R""(하이퍼 파라미터 gamma (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_t__doc__,
    u8R""(the hyperparameter t (read-only))"",
    u8R""(하이퍼 파라미터 t (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_g__doc__,
    u8R""(the hyperparameter alpha_g (read-only))"",
    u8R""(하이퍼 파라미터 alpha_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_l__doc__,
    u8R""(the hyperparameter alpha_l (read-only))"",
    u8R""(하이퍼 파라미터 alpha_l (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_mg__doc__,
    u8R""(the hyperparameter alpha_mg (read-only))"",
    u8R""(하이퍼 파라미터 alpha_mg (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_ml__doc__,
    u8R""(the hyperparameter alpha_ml (read-only))"",
    u8R""(하이퍼 파라미터 alpha_ml (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_eta_g__doc__,
    u8R""(the hyperparameter eta_g (read-only))"",
    u8R""(하이퍼 파라미터 eta_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_eta_l__doc__,
    u8R""(the hyperparameter eta_l (read-only))"",
    u8R""(하이퍼 파라미터 eta_l (읽기전용))"");


/*
    class PA
*/
DOC_SIGNATURE_EN_KO(PA___init____doc__,
    "PAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k1=1, k2=1, alpha=0.1, subalpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Pachinko Allocation(PA) topic model and its implementation is based on following papers:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Pachinko Allocation(PA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.* `k1` : 상위 토픽의 개수, 1 ~ 32767 사이의 정수.
k1 : int
    상위 토픽의 개수, 1 ~ 32767 사이의 정수
k2 : int
    하위 토픽의 개수, 1 ~ 32767 사이의 정수.
alpha : Union[float, Iterable[float]]
    문헌-상위 토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k1` 길이의 `float` 리스트로 입력할 수 있습니다.
subalpha : Union[float, Iterable[float]]
    .. versionadded:: 0.11.0

    상위-하위 토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k2` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    하위 토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(PA_get_topic_words__doc__,
    "get_topic_words(self, sub_topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the sub topic `sub_topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
sub_topic_id : int
    indicating the sub topic, in range [0, `k2`)
)"",
u8R""(하위 토픽 `sub_topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
sub_topic_id : int
    하위 토픽을 가리키는 [0, `k2`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(PA_get_topic_word_dist__doc__,
    "get_topic_word_dist(self, sub_topic_id, normalize=True)",
    u8R""(Return the word distribution of the sub topic `sub_topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current sub topic.

Parameters
----------
sub_topic_id : int
    indicating the sub topic, in range [0, `k2`)
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
)"",
u8R""(하위 토픽 `sub_topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 하위 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
sub_topic_id : int
    하위 토픽을 가리키는 [0, `k2`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_SIGNATURE_EN_KO(PA_get_sub_topics__doc__,
    "get_sub_topics(self, super_topic_id, top_n=10)",
    u8R""(.. versionadded:: 0.1.4

Return the `top_n` sub topics and its probability in a super topic `super_topic_id`.
The return type is a `list` of (subtopic:`int`, probability:`float`).

Parameters
----------
super_topic_id : int
    indicating the super topic, in range [0, `k1`)
)"",
u8R""(.. versionadded:: 0.1.4

상위 토픽 `super_topic_id`에 속하는 상위 `top_n`개의 하위 토픽과 각각의 확률을 반환합니다. 
반환 타입은 (하위토픽:`int`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
super_topic_id : int
    상위 토픽을 가리키는 [0, `k1`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(PA_get_sub_topic_dist__doc__,
    "get_sub_topic_dist(self, super_topic_id, normalize=True)",
    u8R""(Return a distribution of the sub topics in a super topic `super_topic_id`.
The returned value is a `list` that has `k2` fraction numbers indicating probabilities for each sub topic in the current super topic.

Parameters
----------
super_topic_id : int
    indicating the super topic, in range [0, `k1`)
)"",
u8R""(상위 토픽 `super_topic_id`의 하위 토픽 분포를 반환합니다.
반환하는 값은 현재 상위 토픽 내 각각의 하위 토픽들의 발생확률을 나타내는 `k2`개의 소수로 구성된 `list`입니다.

Parameters
----------
super_topic_id : int
    상위 토픽을 가리키는 [0, `k1`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_SIGNATURE_EN_KO(PA_infer__doc__,
    "infer(self, doc, iter=100, tolerance=-1, workers=0, parallel=0, together=False)",
    u8R""(.. versionadded:: 0.5.0

Return the inferred topic distribution and sub-topic distribution from unseen `doc`s.

Parameters
----------
doc : Union[tomotopy.Document, Iterable[tomotopy.Document], tomotopy.utils.Corpus]
    an instance of `tomotopy.Document` or a `list` of instances of `tomotopy.Document` to be inferred by the model.
    It can be acquired from `tomotopy.LDAModel.make_doc` method.

    .. versionchanged:: 0.10.0

        Since version 0.10.0, `infer` can receive a raw corpus instance of `tomotopy.utils.Corpus`. 
        In this case, you don't need to call `make_doc`. `infer` would generate documents bound to the model, estimate its topic distributions and
        return a corpus contains generated documents as the result.
iter : int
    an integer indicating the number of iteration to estimate the distribution of topics of `doc`.
    The higher value will generate a more accuracy result.
tolerance : float
    isn't currently used.
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0
    
    the parallelism scheme for inference. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.
together : bool
    all `doc`s are infered together in one process if True, otherwise each `doc` is infered independently. Its default value is `False`.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    a callable object to manipulate arbitrary keyword arguments for a specific topic model. 
    Available when `doc` is given as an instance of `tomotopy.utils.Corpus`.

Returns
-------
result : Union[Tuple[List[float], List[float]], List[Tuple[List[float], List[float]]], tomotopy.utils.Corpus]
    If `doc` is given as a single `tomotopy.Document`, `result` is a tuple of `List[float]` indicating its topic distribution and `List[float]` indicating its sub-topic distribution.
    
    If `doc` is given as a list of `tomotopy.Document`s, `result` is a list of `List[float]` indicating topic distributions for each document.
    
    If `doc` is given as an instance of `tomotopy.utils.Corpus`, `result` is another instance of `tomotopy.utils.Corpus` which contains infered documents.
    You can get topic distribution for each document using `tomotopy.Document.get_topic_dist` and sub-topic distribution using `tomotopy.Document.get_sub_topic_dist`
log_ll : float
    a list of log-likelihoods for each `doc`s
)"",
u8R""(.. versionadded:: 0.5.0

새로운 문헌인 `doc`에 대해 각각의 주제 분포를 추론하여 반환합니다.
반환 타입은 ((`doc`의 주제 분포, `doc`의 하위 주제 분포), 로그가능도) 또는 ((`doc`의 주제 분포, `doc`의 하위 주제 분포)로 구성된 `list`, 로그가능도)입니다.

Parameters
----------
doc : Union[tomotopy.Document, Iterable[tomotopy.Document]]
    추론에 사용할 `tomotopy.Document`의 인스턴스이거나 이 인스턴스들의 `list`.
    이 인스턴스들은 `tomotopy.LDAModel.make_doc` 메소드를 통해 얻을 수 있습니다.

    .. versionchanged:: 0.10.0

        0.10.0버전부터 `infer`는 `tomotopy.utils.Corpus`의 인스턴스를 직접 입력 받을 수 있습니다. 
        이 경우 `make_doc`를 사용할 필요 없이 `infer`가 직접 모델에 맞춰진 문헌을 생성하고 이를 이용해 토픽 분포를 추정하며,
        결과로 생성된 문헌들이 포함된 `tomotopy.utils.Corpus`를 반환합니다.
iter : int
    `doc`의 주제 분포를 추론하기 위해 학습을 반복할 횟수입니다.
    이 값이 클 수록 더 정확한 결과를 낼 수 있습니다.
tolerance : float
    현재는 사용되지 않음
workers : int
    깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
    만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
parallel : Union[int, tomotopy.ParallelScheme]
    .. versionadded:: 0.5.0

    추론에 사용할 병렬화 방법. 기본값은 ParallelScheme.DEFAULT로 이는 모델에 따라 최적의 방법을 tomotopy가 알아서 선택하도록 합니다.
together : bool
    이 값이 True인 경우 입력한 `doc` 문헌들을 한 번에 모델에 넣고 추론을 진행합니다.
    False인 경우 각각의 문헌들을 별도로 모델에 넣어 추론합니다. 기본값은 `False`입니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.10.0
    
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체.
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우에만 사용 가능합니다.

Returns
-------
result : Union[Tuple[List[float], List[float]], List[Tuple[List[float], List[float]]], tomotopy.utils.Corpus]
    `doc`이 `tomotopy.Document`로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`와 하위 토픽 분포를 나타내는 `List[float]`의 tuple입니다.
    
    `doc`이 `tomotopy.Document`의 list로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`와 하위 토픽 분포를 나타내는 `List[float]`의 tuple의 list입니다.
    
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우, `result`는 추론된 결과 문서들을 담고 있는, `tomotopy.utils.Corpus`의 새로운 인스턴스입니다.
    각 문헌별 토픽 분포를 얻기 위해서는 `tomotopy.Document.get_topic_dist`, 하위 토픽 분포를 얻기 위해서는 `tomotopy.Document.get_sub_topic_dist`를 사용하면 됩니다.
log_ll : List[float]
    각 문헌별 로그 가능도의 리스트
)"");

DOC_SIGNATURE_EN_KO(PA_get_count_by_super_topic__doc__,
    "get_count_by_super_topic(self)",
    u8R""(Return the number of words allocated to each super-topic.

.. versionadded:: 0.9.0)"",
u8R""(각각의 상위 토픽에 할당된 단어의 개수를 `list`형태로 반환합니다.

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(PA_k1__doc__,
    u8R""(k1, the number of super topics (read-only))"",
    u8R""(k1, 상위 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(PA_k2__doc__,
    u8R""(k2, the number of sub topics (read-only))"",
    u8R""(k2, 하위 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(PA_alpha__doc__,
    u8R""(Dirichlet prior on the per-document super topic distributions in shape `[k1]` (read-only)

.. versionadded:: 0.9.0)"",
u8R""(문헌의 상위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1]` 모양 (읽기전용)

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(PA_subalpha__doc__,
    u8R""(Dirichlet prior on the sub topic distributions for each super topic in shape `[k1, k2]` (read-only)

.. versionadded:: 0.9.0)"",
u8R""(상위 토픽의 하위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1, k2]` 모양 (읽기전용)

.. versionadded:: 0.9.0)"");


/*
    class HPA
*/
DOC_SIGNATURE_EN_KO(HPA___init____doc__,
    "HPAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k1=1, k2=1, alpha=0.1, subalpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Hierarchical Pachinko Allocation(HPA) topic model and its implementation is based on following papers:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    .. versionadded:: 0.2.0
    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Hierarchical Pachinko Allocation(HPA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.* `k1` : 상위 토픽의 개수, 1 ~ 32767 사이의 정수.
k1 : int
    상위 토픽의 개수, 1 ~ 32767 사이의 정수
k2 : int
    하위 토픽의 개수, 1 ~ 32767 사이의 정수.
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k1 + 1` 길이의 `float` 리스트로 입력할 수 있습니다.
subalpha : Union[float, Iterable[float]]
    .. versionadded:: 0.11.0

    상위-하위 토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k2 + 1` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(HPA_get_topic_words__doc__,
    "get_topic_words(self, topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    0 indicates the top topic, 
    a number in range [1, 1 + `k1`) indicates a super topic and
    a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    0일 경우 최상위 토픽을 가리키며,
    [1, 1 + `k1`) 범위의 정수는 상위 토픽을,
    [1 + `k1`, 1 + `k1` + `k2`) 범위의 정수는 하위 토픽을 가리킵니다.
)"");

DOC_SIGNATURE_EN_KO(HPA_get_topic_word_dist__doc__,
    "get_topic_word_dist(self, topic_id, normalize=True)",
    u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in current topic.

Parameters
----------
topic_id : int
    0 indicates the top topic, 
    a number in range [1, 1 + `k1`) indicates a super topic and
    a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
normalize : bool
    .. versionadded:: 0.11.0

    If True, it returns the probability distribution with the sum being 1. Otherwise it returns the distribution of raw values.
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 하위 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    0일 경우 최상위 토픽을 가리키며,
    [1, 1 + `k1`) 범위의 정수는 상위 토픽을,
    [1 + `k1`, 1 + `k1` + `k2`) 범위의 정수는 하위 토픽을 가리킵니다.
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_VARIABLE_EN_KO(HPA_alpha__doc__,
    u8R""(Dirichlet prior on the per-document super topic distributions in shape `[k1 + 1]`. 
Its element 0 indicates the prior to the top topic and elements 1 ~ k1 indicates ones to the super topics. (read-only)

.. versionadded:: 0.9.0)"",
u8R""(문헌의 상위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1 + 1]` 모양.
0번째 요소는 최상위 토픽을 가리키며, 1 ~ k1번째가 상위 토픽을 가리킨다. (읽기전용)

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(HPA_subalpha__doc__,
    u8R""(Dirichlet prior on the sub topic distributions for each super topic in shape `[k1, k2 + 1]`.
Its `[x, 0]` element indicates the prior to the super topic `x` 
and `[x, 1 ~ k2]` elements indicate ones to the sub topics in the super topic `x`. (read-only)

.. versionadded:: 0.9.0)"",
u8R""(상위 토픽의 하위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1, k2 + 1]` 모양.
`[x, 0]` 요소는 상위 토픽 `x`를 가리키며, `[x, 1 ~ k2]` 요소는 상위 토픽 `x` 내의 하위 토픽들을 가리킨다. (읽기전용)

.. versionadded:: 0.9.0)"");


/*
    class CT
*/

DOC_SIGNATURE_EN_KO(CT___init____doc__,
    "CTModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, smoothing_alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(.. versionadded:: 0.2.0
This type provides Correlated Topic Model (CTM) and its implementation is based on following papers:
	
> * Blei, D., & Lafferty, J. (2006). Correlated topic models. Advances in neural information processing systems, 18, 147.
> * Mimno, D., Wallach, H., & McCallum, A. (2008, December). Gibbs sampling for logistic normal topic models with graph-based priors. In NIPS Workshop on Analyzing Graphs (Vol. 61).

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(.. versionadded:: 0.2.0
이 타입은 Correlated Topic Model (CTM)의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Blei, D., & Lafferty, J. (2006). Correlated topic models. Advances in neural information processing systems, 18, 147.
> * Mimno, D., Wallach, H., & McCallum, A. (2008, December). Gibbs sampling for logistic normal topic models with graph-based priors. In NIPS Workshop on Analyzing Graphs (Vol. 61).

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 사이의 정수
smoothing_alpha : Union[float, Iterable[float]]
    토픽 개수가 0이 되는걸 방지하는 평탄화 계수, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(CT_get_correlations__doc__,
    "get_correlations(self, topic_id=None)",
    u8R""(Return correlations between the topic `topic_id` and other topics.
The returned value is a `list` of `float`s of size `tomotopy.LDAModel.k`.

Parameters
----------
topic_id : Union[int, None]
    an integer in range [0, `k`), indicating the topic
    
    If omitted, the whole correlation matrix is returned.
)"",
u8R""(토픽 `topic_id`와 나머지 토픽들 간의 상관관계를 반환합니다.
반환값은 `tomotopy.LDAModel.k` 길이의 `float`의 `list`입니다.

Parameters
----------
topic_id : Union[int, None]
    토픽을 지정하는 [0, `k`), 범위의 정수

    생략 시 상관계수 행렬 전체가 반환됩니다.
)"");

DOC_VARIABLE_EN_KO(CT_num_beta_sample__doc__,
    u8R""(the number of times to sample beta parameters, default value is 10.

CTModel samples `num_beta_sample` beta parameters for each document. 
The more beta it samples, the more accurate the distribution will be, but the longer time it takes to learn. 
If you have a small number of documents in your model, keeping this value larger will help you get better result.
)"",
u8R""(beta 파라미터를 표집하는 횟수, 기본값은 10.

CTModel은 각 문헌마다 총 `num_beta_sample` 개수의 beta 파라미터를 표집합니다.
beta 파라미터를 더 많이 표집할 수록, 전체 분포는 정교해지지만 학습 시간이 더 많이 걸립니다.
만약 모형 내에 문헌의 개수가 적은 경우 이 값을 크게하면 더 정확한 결과를 얻을 수 있습니다.
)"");

DOC_VARIABLE_EN_KO(CT_num_tmn_sample__doc__,
    u8R""(the number of iterations for sampling Truncated Multivariate Normal distribution, default value is 5.

If your model shows biased topic correlations, increasing this value may be helpful.)"",
u8R""(절단된 다변수 정규분포에서 표본을 추출하기 위한 반복 횟수, 기본값은 5.

만약 결과에서 토픽 간 상관관계가 편향되게 나올 경우 이 값을 키우면 편향을 해소하는 데에 도움이 될 수 있습니다.
)"");

DOC_VARIABLE_EN_KO(CT_prior_mean__doc__,
    u8R""(the mean of prior logistic-normal distribution for the topic distribution (read-only))"",
    u8R""(토픽의 사전 분포인 로지스틱 정규 분포의 평균 벡터 (읽기전용))"");

DOC_VARIABLE_EN_KO(CT_prior_cov__doc__,
    u8R""(the covariance matrix of prior logistic-normal distribution the for topic distribution (read-only))"",
    u8R""(토픽의 사전 분포인 로지스틱 정규 분포의 공분산 행렬 (읽기전용))"");

DOC_VARIABLE_EN_KO(CT_alpha__doc__,
    u8R""(This property is not available in `CTModel`. Use `CTModel.prior_mean` and `CTModel.prior_cov` instead.

.. versionadded:: 0.9.1)"",
u8R""(이 프로퍼티는 `CTModel`에서 사용불가합니다. 대신 `CTModel.prior_mean`와 `CTModel.prior_cov`를 사용하십시오.

.. versionadded:: 0.9.1)"");

/*
    class SLDA
*/
DOC_SIGNATURE_EN_KO(SLDA___init____doc__,
    "SLDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, vars='', alpha=0.1, eta=0.01, mu=[], nu_sq=[], glm_param=[], seed=None, corpus=None, transform=None)",
    u8R""(This type provides supervised Latent Dirichlet Allocation(sLDA) topic model and its implementation is based on following papers:
	
> * Mcauliffe, J. D., & Blei, D. M. (2008). Supervised topic models. In Advances in neural information processing systems (pp. 121-128).
> * Python version implementation using Gibbs sampling : https://github.com/Savvysherpa/slda

.. versionadded:: 0.2.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 supervised Latent Dirichlet Allocation(sLDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Mcauliffe, J. D., & Blei, D. M. (2008). Supervised topic models. In Advances in neural information processing systems (pp. 121-128).
> * Python version implementation using Gibbs sampling : https://github.com/Savvysherpa/slda

.. versionadded:: 0.2.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 사이의 정수
vars : Iterable[str]
    응답변수의 종류를 지정합니다.
    `vars`의 길이는 모형이 사용하는 응답 변수의 개수를 결정하며, `vars`의 요소는 각 응답 변수의 종류를 결정합니다.
    사용가능한 종류는 다음과 같습니다:
    
    > * 'l': 선형 변수 (아무 실수 값이나 가능)
    > * 'b': 이진 변수 (0 혹은 1만 가능)
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
mu : Union[float, Iterable[float]]
    회귀 계수의 평균값, 기본값은 0
nu_sq : Union[float, Iterable[float]]
    회귀 계수의 분산값, 기본값은 1
glm_param : Union[float, Iterable[float]]
    일반화 선형 모형에서 사용될 파라미터, 기본값은 1
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(SLDA_add_doc__doc__,
    "add_doc(self, words, y=[], ignore_empty_words=True)",
    u8R""(Add a new document into the model instance with response variables `y` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
y : Iterable[float]
    response variables of this document. 
    The length of `y` must be equal to the number of response variables of the model (`tomotopy.SLDAModel.f`).
    
    .. versionchanged:: 0.5.1
    
        If you have a missing value, you can set the item as `NaN`. Documents with `NaN` variables are included in modeling topics, but excluded from regression.
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 응답 변수 `y`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
y : Iterable[float]
    문헌의 응답 변수로 쓰일 `float`의 `list`. `y`의 길이는 모델의 응답 변수의 개수인 `tomotopy.SLDAModel.f`와 일치해야 합니다.
    
    .. versionchanged:: 0.5.1
    
        만약 결측값이 있을 경우, 해당 항목을 `NaN`으로 설정할 수 있습니다. 이 경우 `NaN`값을 가진 문헌은 토픽을 모델링하는 데에는 포함되지만, 응답 변수 회귀에서는 제외됩니다.
)"");

DOC_SIGNATURE_EN_KO(SLDA_make_doc__doc__,
    "make_doc(self, words, y=[])",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and response variables `y` that can be used for `tomotopy.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
y : Iterable[float]
    response variables of this document. 
    The length of `y` doesn't have to be equal to the number of response variables of the model (`tomotopy.SLDAModel.f`).
    If the length of `y` is shorter than `tomotopy.SLDAModel.f`, missing values are automatically filled with `NaN`.
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
y : Iterable[float]
    문헌의 응답 변수로 쓰일 `float`의 `list`. 
    `y`의 길이는 모델의 응답 변수의 개수인 `tomotopy.SLDAModel.f`와 꼭 일치할 필요는 없습니다.
    `y`의 길이가 `tomotopy.SLDAModel.f`보다 짧을 경우, 모자란 값들은 자동으로 `NaN`으로 채워집니다.
)"");

DOC_SIGNATURE_EN_KO(SLDA_get_regression_coef__doc__,
    "get_regression_coef(self, var_id=None)",
    u8R""(Return the regression coefficient of the response variable `var_id`.

Parameters
----------
var_id : int
    indicating the reponse variable, in range [0, `f`)

    If omitted, the whole regression coefficients with shape `[f, k]` are returned.
)"",
u8R""(응답 변수 `var_id`의 회귀 계수를 반환합니다.

Parameters
----------
var_id : int
    응답 변수를 지정하는 [0, `f`) 범위의 정수

    생략시, `[f, k]` 모양의 전체 회귀 계수가 반환됩니다.
)"");

DOC_SIGNATURE_EN_KO(SLDA_get_var_type__doc__,
    "get_var_type(self, var_id)",
    u8R""(Return the type of the response variable `var_id`. 'l' means linear variable, 'b' means binary variable.)"",
    u8R""(응답 변수 `var_id`의 종류를 반환합니다. 'l'은 선형 변수, 'b'는 이진 변수를 뜻합니다.)"");

DOC_SIGNATURE_EN_KO(SLDA_estimate__doc__,
    "estimate(self, doc)",
    u8R""(Return the estimated response variable for `doc`.
If `doc` is an unseen document instance which is generated by `tomotopy.SLDAModel.make_doc` method, it should be inferred by `tomotopy.LDAModel.infer` method first.

Parameters
----------
doc : tomotopy.Document
    an instance of document or a list of them to be used for estimating response variables
)"",
u8R""(`doc`의 추정된 응답 변수를 반환합니다.
만약 `doc`이 `tomotopy.SLDAModel.make_doc`에 의해 생성된 인스턴스라면, 먼저 `tomotopy.LDAModel.infer`를 통해 토픽 추론을 실시한 다음 이 메소드를 사용해야 합니다.

Parameters
----------
doc : tomotopy.Document
    응답 변수를 추정하려하는 문헌의 인스턴스 혹은 인스턴스들의 list
)"");

DOC_VARIABLE_EN_KO(SLDA_f__doc__,
    u8R""(the number of response variables (read-only))"",
    u8R""(응답 변수의 개수 (읽기전용))"");

/*
    class LLDA
*/
DOC_SIGNATURE_EN_KO(LLDA___init____doc__,
    "LLDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Labeled LDA(L-LDA) topic model and its implementation is based on following papers:
	
> * Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.

.. versionadded:: 0.3.0

.. deprecated:: 0.11.0
    Use `tomotopy.PLDAModel` instead.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Labeled LDA(L-LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.

.. versionadded:: 0.3.0

.. deprecated:: 0.11.0
    `tomotopy.PLDAModel`를 대신 사용하세요.

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 범위의 정수.
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(LLDA_add_doc__doc__,
    "add_doc(self, words, labels=[], ignore_empty_words=True)",
    u8R""(Add a new document into the model instance with `labels` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
labels : Iterable[str]
    labels of the document
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 `labels`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
labels : Iterable[str]
    문헌의 레이블 리스트
)"");

DOC_SIGNATURE_EN_KO(LLDA_make_doc__doc__,
    "make_doc(self, words, labels=[])",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `labels` that can be used for `tomotopy.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iteratable of `str`
labels : Iterable[str]
    labels of the document
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
labels : Iterable[str]
    문헌의 레이블 리스트
)"");

DOC_SIGNATURE_EN_KO(LLDA_get_topic_words__doc__,
    "get_topic_words(self, topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    Integers in the range [0, `l`), where `l` is the number of total labels, represent a topic that belongs to the corresponding label.
    The label name can be found by looking up `tomotopy.LLDAModel.topic_label_dict`.
    Integers in the range [`l`, `k`) represent a latent topic which doesn't belongs to the any labels.
    
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    전체 레이블의 개수를 `l`이라고 할 때, [0, `l`) 범위의 정수는 각각의 레이블에 해당하는 토픽을 가리킵니다. 
    해당 토픽의 레이블 이름은 `tomotopy.LLDAModel.topic_label_dict`을 열람하여 확인할 수 있습니다.
    [`l`, `k`) 범위의 정수는 어느 레이블에도 속하지 않는 잠재 토픽을 가리킵니다.
)"");


DOC_VARIABLE_EN_KO(LLDA_topic_label_dict__doc__,
    u8R""(a dictionary of topic labels in type `tomotopy.Dictionary` (read-only))"",
    u8R""(`tomotopy.Dictionary` 타입의 토픽 레이블 사전 (읽기전용))"");

/*
    class PLDA
*/
DOC_SIGNATURE_EN_KO(PLDA___init____doc__,
    "PLDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, latent_topics=0, topics_per_label=1, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Partially Labeled LDA(PLDA) topic model and its implementation is based on following papers:
	
> * Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.

.. versionadded:: 0.4.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(이 타입은 Partially Labeled LDA(PLDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.

.. versionadded:: 0.4.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
latent_topics : int
    모든 문헌에 공유되는 잠재 토픽의 개수, 1 ~ 32767 사이의 정수.
topics_per_label : int
    레이블별 토픽의 개수, 1 ~ 32767 사이의 정수.
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");


DOC_SIGNATURE_EN_KO(PLDA_get_topic_words__doc__,
    "get_topic_words(self, topic_id, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    Integers in the range [0, `l` * `topics_per_label`), where `l` is the number of total labels, represent a topic that belongs to the corresponding label.
    The label name can be found by looking up `tomotopy.PLDAModel.topic_label_dict`.
    Integers in the range [`l` * `topics_per_label`, `l` * `topics_per_label` + `latent_topics`) represent a latent topic which doesn't belongs to the any labels.
    
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    전체 레이블의 개수를 `l`이라고 할 때, [0, `l` * `topics_per_label`) 범위의 정수는 각각의 레이블에 해당하는 토픽을 가리킵니다. 
    해당 토픽의 레이블 이름은 `tomotopy.PLDAModel.topic_label_dict`을 열람하여 확인할 수 있습니다.
    [`l` * `topics_per_label`, `l` * `topics_per_label` + `latent_topics`) 범위의 정수는 어느 레이블에도 속하지 않는 잠재 토픽을 가리킵니다.
)"");


DOC_VARIABLE_EN_KO(PLDA_topic_label_dict__doc__,
    u8R""(a dictionary of topic labels in type `tomotopy.Dictionary` (read-only))"",
    u8R""(`tomotopy.Dictionary` 타입의 토픽 레이블 사전 (읽기전용))"");

DOC_VARIABLE_EN_KO(PLDA_latent_topics__doc__,
    u8R""(the number of latent topics (read-only))"",
    u8R""(잠재 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(PLDA_topics_per_label__doc__,
    u8R""(the number of topics per label (read-only))"",
    u8R""(레이블별 토픽의 개수 (읽기전용))"");

/*
    class HLDA
*/
DOC_SIGNATURE_EN_KO(HLDA___init____doc__,
    "HLDAModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, depth=2, alpha=0.1, eta=0.01, gamma=0.1, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Hierarchical LDA topic model and its implementation is based on following papers:

> * Griffiths, T. L., Jordan, M. I., Tenenbaum, J. B., & Blei, D. M. (2004). Hierarchical topic models and the nested Chinese restaurant process. In Advances in neural information processing systems (pp. 17-24).

.. versionadded:: 0.4.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    .. versionadded:: 0.6.0

    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int    
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
depth : int
    the maximum depth level of hierarchy between 2 ~ 32767
alpha : Union[float, Iterable[float]]
    hyperparameter of Dirichlet distribution for document-depth level, given as a single `float` in case of symmetric prior and as a list with length `depth` of `float` in case of asymmetric prior.
eta : float
    hyperparameter of Dirichlet distribution for topic-word
gamma : float
    concentration coeficient of Dirichlet Process
seed : int
    random seed. default value is a random number from `std::random_device{}` in C++
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    a list of documents to be added into the model
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    a callable object to manipulate arbitrary keyword arguments for a specific topic model
)"",
u8R""(이 타입은 Hierarchical LDA 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Griffiths, T. L., Jordan, M. I., Tenenbaum, J. B., & Blei, D. M. (2004). Hierarchical topic models and the nested Chinese restaurant process. In Advances in neural information processing systems (pp. 17-24).

.. versionadded:: 0.4.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    .. versionadded:: 0.6.0

    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    .. versionadded:: 0.2.0    
    
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
depth : int
    토픽 계층의 깊이를 지정하는 2 ~ 32767 범위의 정수.
alpha : Union[float, Iterable[float]]
    문헌-계층 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `depth` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
gamma : float
    Dirichlet Process의 집중 계수
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    .. versionadded:: 0.6.0

    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    .. versionadded:: 0.6.0

    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(HLDA_is_live_topic__doc__,
    "is_live_topic(self, topic_id)",
    u8R""(Return `True` if the topic `topic_id` is alive, otherwise return `False`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id`가 유효한 토픽을 가리키는 경우 `True`, 아닌 경우 `False`를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(HLDA_num_docs_of_topic__doc__,
    "num_docs_of_topic(self, topic_id)",
    u8R""(Return the number of documents belonging to a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id` 토픽에 속하는 문헌의 개수를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(HLDA_level__doc__,
    "level(self, topic_id)",
    u8R""(Return the level of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id` 토픽의 레벨을 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(HLDA_parent_topic__doc__,
    "parent_topic(self, topic_id)",
    u8R""(Return the topic ID of parent of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id` 토픽의 부모 토픽의 ID를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(HLDA_children_topics__doc__,
    "children_topics(self, topic_id)",
    u8R""(Return a list of topic IDs with children of a topic `topic_id`.

Parameters
----------
topic_id : int
    an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id` 토픽의 자식 토픽들의 ID를 list로 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_VARIABLE_EN_KO(HLDA_gamma__doc__,
    u8R""(the hyperparameter gamma (read-only))"",
    u8R""(하이퍼 파라미터 gamma (읽기전용))"");

DOC_VARIABLE_EN_KO(HLDA_live_k__doc__,
    u8R""(the number of alive topics (read-only))"",
    u8R""(현재 모델 내의 유효한 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(HLDA_depth__doc__,
    u8R""(the number of depth (read-only))"",
    u8R""(현재 모델의 총 깊이 (읽기전용))"");

/*
    class DT
*/
DOC_SIGNATURE_EN_KO(DT___init____doc__,
    "DTModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, t=1, alpha_var=0.1, eta_var=0.1, phi_var=0.1, lr_a=0.01, lr_b=0.1, lr_c=0.55, seed=None, corpus=None, transform=None)",
    u8R""(This type provides Dynamic Topic model and its implementation is based on following papers:

> * Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).
> * Bhadury, A., Chen, J., Zhu, J., & Liu, S. (2016, April). Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web (pp. 381-390).
> https://github.com/Arnie0426/FastDTM

.. versionadded:: 0.7.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
    The default value is 0, which means no top words are removed.
k : int
    the number of topics between 1 ~ 32767
t : int
    the number of timpoints
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
)"",
u8R""(이 타입은 Dynamic Topic Model의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).
> * Bhadury, A., Chen, J., Zhu, J., & Liu, S. (2016, April). Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web (pp. 381-390).
> https://github.com/Arnie0426/FastDTM

.. versionadded:: 0.7.0

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 범위의 정수.
t : int
    시점의 개수
alpha_var : float
    alpha 파라미터(시점별 토픽 분포)의 전이 분산
eta_var : float
    eta 파라미터(문헌별 토픽 분포)의 alpha로부터의 분산
phi_var : float
    phi 파라미터(토픽별 단어 분포)의 전이 분산
lr_a : float
    SGLD의 스텝 크기 `e_i = a * (b + i) ^ (-c)` 계산하는데 사용되는 0보다 큰 `a`값
lr_b : float
    SGLD의 스텝 크기 `e_i = a * (b + i) ^ (-c)` 계산하는데 사용되는 0 이상의 `b`값
lr_c : float
    SGLD의 스텝 크기 `e_i = a * (b + i) ^ (-c)` 계산하는데 사용되는 (0.5, 1] 범위의 `c`값 
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");

DOC_SIGNATURE_EN_KO(DT_add_doc__doc__,
    "add_doc(self, words, timepoint=0, ignore_empty_words=True)",
    u8R""(Add a new document into the model instance with `timepoint` and return an index of the inserted document.

Parameters
----------
words : Iterable[str]
    an iterable of `str`
timepoint : int
    an integer with range [0, `t`)
ignore_empty_words : bool
    If `True`, empty `words` doesn't raise exception and makes the method return None.
)"",
u8R""(현재 모델에 `timepoint` 시점의 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(DT_make_doc__doc__,
    "make_doc(self, words, timepoint=0)",
    u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `timepoint` that can be used for `tomotopy.LDAModel.infer` method.

Parameters
----------
words : Iterable[str]
    an iteratable of `str`
timepoint : int
    an integer with range [0, `t`)
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(DT_get_alpha__doc__,
    "get_alpha(self, timepoint)",
    u8R""(Return a `list` of alpha parameters for `timepoint`.

Parameters
----------
timepoint : int
    an integer with range [0, `t`)
)"",
u8R""(`timepoint` 시점에 대한 alpha 파라미터의 리스트를 반환합니다.

Parameters
----------
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(DT_get_phi__doc__,
    "get_phi(self, timepoint, topic_id)",
    u8R""(Return a `list` of phi parameters for `timepoint` and `topic_id`.

Parameters
----------
timepoint : int
    an integer with range [0, `t`)
topic_id : int
    an integer with range [0, `k`)
)"",
u8R""(`timepoint` 시점의 `topic_id`에 대한 phi 파라미터의 리스트를 반환합니다.

Parameters
----------
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
topic_id : int
    토픽을 나타내는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(DT_get_topic_words__doc__,
    "get_topic_words(self, topic_id, timepoint, top_n=10)",
    u8R""(Return the `top_n` words and its probability in the topic `topic_id` with `timepoint`. 
The return type is a `list` of (word:`str`, probability:`float`).

Parameters
----------
topic_id : int
    an integer in range [0, `k`), indicating the topic
timepoint : int
	an integer in range [0, `t`), indicating the timepoint
)"",
u8R""(시점 `timepoint`의 토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
timepoint : int
	시점을 가리키는 [0, `t`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(DT_get_topic_word_dist__doc__,
    "get_topic_word_dist(self, topic_id, timepoint, normalize=True)",
    u8R""(Return the word distribution of the topic `topic_id` with `timepoint`.
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
)"",
u8R""(시점 `timepoint`의 토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
timepoint : int
	시점을 가리키는 [0, `t`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
)"");

DOC_SIGNATURE_EN_KO(DT_get_count_by_topics__doc__,
    "get_count_by_topics(self)",
    u8R""(Return the number of words allocated to each timepoint and topic in the shape `[num_timepoints, k]`.

.. versionadded:: 0.9.0)"",
u8R""(각각의 시점과 토픽에 할당된 단어의 개수를 `[num_timepoints, k]` 모양으로 반환합니다.

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(DT_lr_a__doc__,
    u8R""(parameter `a` greater than zero for SGLD step size (e_i = a * (b + i) ^ -c))"",
    u8R""(SGLD의 스텝 크기를 결정하는 0보다 큰 파라미터 `a` (e_i = a * (b + i) ^ -c))"");

DOC_VARIABLE_EN_KO(DT_lr_b__doc__,
    u8R""(parameter `b` greater than zero or equal to zero for SGLD step size (e_i = a * (b + i) ^ -c))"",
    u8R""(SGLD의 스텝 크기를 결정하는 0 이상의 파라미터 `b` (e_i = a * (b + i) ^ -c))"");

DOC_VARIABLE_EN_KO(DT_lr_c__doc__,
    u8R""(parameter `c` with range (0.5, 1] for SGLD step size (e_i = a * (b + i) ^ -c))"",
    u8R""(SGLD의 스텝 크기를 결정하는 (0.5, 1] 범위의 파라미터 `c` (e_i = a * (b + i) ^ -c))"");

DOC_VARIABLE_EN_KO(DT_num_timepoints__doc__,
    u8R""(the number of timepoints of the model (read-only))"",
    u8R""(모델의 시점 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(DT_num_docs_by_timepoint__doc__,
    u8R""(the number of documents in the model by timepoint (read-only))"",
    u8R""(각 시점별 모델 내 문헌 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(DT_alpha__doc__,
    u8R""(per-document topic distribution in the shape `[num_timepoints, k]` (read-only)

.. versionadded:: 0.9.0)"",
u8R""(문헌별 토픽 분포, `[num_timepoints, k]` 모양 (읽기전용)

.. versionadded:: 0.9.0)"");

DOC_VARIABLE_EN_KO(DT_eta__doc__,
    u8R""(This property is not available in `DTModel`. Use `DTModel.docs[x].eta` instead.

.. versionadded:: 0.9.0)"",
u8R""(이 프로퍼티는 `DTModel`에서 사용불가합니다. 대신 `DTModel.docs[x].eta`를 사용하십시오.

.. versionadded:: 0.9.0)"");


/*
    class PT
*/

DOC_SIGNATURE_EN_KO(PT___init____doc__,
    "PTModel(tw=TermWeight.ONE, min_cf=0, min_df=0, rm_top=0, k=1, p=None, alpha=0.1, eta=0.01, seed=None, corpus=None, transform=None)",
    u8R""(.. versionadded:: 0.11.0
This type provides Pseudo-document based Topic Model (PTM) and its implementation is based on following papers:
	
> * Zuo, Y., Wu, J., Zhang, H., Lin, H., Wang, F., Xu, K., & Xiong, H. (2016, August). Topic modeling of short texts: A pseudo-document view. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 2105-2114).

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
min_cf : int
    minimum collection frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
    The default value is 0, which means no words are excluded.
min_df : int
    minimum document frequency of words. Words with a smaller document frequency than `min_df` are excluded from the model.
    The default value is 0, which means no words are excluded
rm_top : int
    the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more.
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
)"",
u8R""(.. versionadded:: 0.11.0
이 타입은 Pseudo-document based Topic Model (PTM)의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Zuo, Y., Wu, J., Zhang, H., Lin, H., Wang, F., Xu, K., & Xiong, H. (2016, August). Topic modeling of short texts: A pseudo-document view. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 2105-2114).

Parameters
----------
tw : Union[int, tomotopy.TermWeight]
    용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
min_cf : int
    단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
min_df : int
    단어의 최소 문헌 빈도. 출현한 문헌 숫자가 `min_df`보다 작은 단어들은 모델에서 제외시킵니다.
    기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
rm_top : int
    제거될 최상위 빈도 단어의 개수. 만약 너무 흔한 단어가 토픽 모델 상위 결과에 등장해 이를 제거하고 싶은 경우, 이 값을 1 이상의 수로 설정하십시오.
    기본값은 0으로, 이 경우 최상위 빈도 단어는 전혀 제거되지 않습니다.
k : int
    토픽의 개수, 1 ~ 32767 사이의 정수
p : int
    가상 문헌의 개수
    ..versionchanged:: 0.12.2
        기본값이 `10 * k`로 변경되었습니다.
alpha : Union[float, Iterable[float]]
    문헌-토픽 디리클레 분포의 하이퍼 파라미터, 대칭일 경우 `float`값 하나로, 비대칭일 경우 `k` 길이의 `float` 리스트로 입력할 수 있습니다.
eta : float
    토픽-단어 디리클레 분포의 하이퍼 파라미터
seed : int
    난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
    이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
corpus : tomotopy.utils.Corpus
    토픽 모델에 추가될 문헌들의 집합을 지정합니다.
transform : Callable[dict, dict]
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
)"");


DOC_VARIABLE_EN_KO(PT_p__doc__,
    u8R""(the number of pseudo documents (read-only)

.. versionadded:: 0.11.0)"",
u8R""(가상 문헌의 개수 (읽기전용)

.. versionadded:: 0.11.0)"");
