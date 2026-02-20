"""Korean documentation overrides for tomotopy.

This module is loaded when TOMOTOPY_LANG=kr environment variable is set.
"""

_MODULE_DOCS = {
    'tomotopy': '''`tomotopy` 패키지는 Python에서 사용가능한 다양한 토픽 모델링 타입과 함수를 제공합니다.
내부 모듈은 c++로 작성되었기 때문에 빠른 속도를 자랑합니다.

.. include:: ./documentation.kr.rst
''',
    'tomotopy.utils': '''`tomotopy.utils` 서브모듈은 토픽 모델링에 유용한 여러 유틸리티를 제공합니다. 
`tomotopy.utils.Corpus` 클래스는 대량의 문헌을 관리할 수 있게 돕습니다. `Corpus`에 입력된 문헌들은 다양한 토픽 모델에 바로 입력될 수 있습니다.
또한 코퍼스 전처리 결과를 파일에 저장함으로써 필요에 따라 다시 코퍼스를 파일에서 읽어들여 원하는 토픽 모델에 입력할 수 있습니다.
''',
    'tomotopy.coherence': '''..versionadded:: 0.10.0

이 모듈은 다음 논문에 의거한 토픽 coherence 계산법을 제공합니다:

> * Röder, M., Both, A., & Hinneburg, A. (2015, February). Exploring the space of topic coherence measures. In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399-408).
> http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
> https://github.com/dice-group/Palmetto
''',
    'tomotopy.label': '''
`tomotopy.label` 서브모듈은 자동 토픽 라벨링 기법을 제공합니다.
아래에 나온 코드처럼 간단한 작업을 통해 토픽 모델의 결과에 이름을 붙일 수 있습니다. 그 결과는 코드 하단에 첨부되어 있습니다.

.. include:: ./auto_labeling_code.rst
''',
}

_DOCS = {
    # Enums
    'isa': '현재 로드된 모듈이 어떤 SIMD 명령어 세트를 사용하는지 표시합니다. \n이 값은 `\'avx512\'`, `\'avx2\'`, `\'sse2\'`, `\'none\'` 중 하나입니다.',
    'TermWeight': '''용어 가중치 기법을 선택하는 데에 사용되는 열거형입니다. 여기에 제시된 용어 가중치 기법들은 다음 논문을 바탕으로 하였습니다:

> * Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.

총 3가지 가중치 기법을 사용할 수 있으며 기본값은 ONE입니다. 기본값뿐만 아니라 다른 모든 기법들도 `tomotopy`의 모든 토픽 모델에 사용할 수 있습니다. ''',
    'TermWeight.ONE': '모든 용어를 동일하게 간주합니다. (기본값)',
    'TermWeight.IDF': '''역문헌빈도(IDF)를 가중치로 사용합니다.

따라서 모든 문헌에 거의 골고루 등장하는 용어의 경우 낮은 가중치를 가지게 되며, 
소수의 특정 문헌에만 집중적으로 등장하는 용어의 경우 높은 가중치를 가지게 됩니다.''',
    'TermWeight.PMI': '점별 상호정보량(PMI)을 가중치로 사용합니다.',
    'ParallelScheme': '병렬화 기법을 선택하는 데에 사용되는 열거형입니다. 총 3가지 기법을 사용할 수 있으나, 모든 모델이 아래의 기법을 전부 지원하지는 않습니다.',
    'ParallelScheme.DEFAULT': 'tomotopy가 모델에 따라 적합한 병럴화 기법을 선택하도록 합니다. 이 값이 기본값입니다.',
    'ParallelScheme.NONE': '깁스 샘플링에 병렬화 기법을 사용하지 않습니다. 깁스 샘플링을 제외한 다른 연산들은 여전히 병렬로 처리될 수 있습니다.',
    'ParallelScheme.COPY_MERGE': '''AD-LDA에서 제안된 복사 후 합치기 알고리즘을 사용합니다. 이는 작업자 수에 비례해 메모리를 소모합니다. 
작업자 수가 적거나, 토픽 개수 혹은 어휘 집합의 크기가 작을 때 유리합니다.
0.5버전 이전까지는 모든 모델은 이 알고리즘을 기본으로 사용했습니다.

> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.''',
    'ParallelScheme.PARTITION': '''PCGS에서 제안된 분할 샘플링 알고리즘을 사용합니다. 작업자 수에 관계없이 단일 스레드 알고리즘에 비해 2배의 메모리만 소모합니다.
작업자 수가 많거나, 토픽 개수 혹은 어휘 집합의 크기가 클 때 유리합니다.

> * Yan, F., Xu, N., & Qi, Y. (2009). Parallel inference for latent dirichlet allocation on graphics processing units. In Advances in neural information processing systems (pp. 2134-2142).''',

    # Coherence enums
    'ProbEstimation': '\n논문에 제시된 probability estimator를 위한 열거형\n',
    'Segmentation': '\n논문에 제시된 segmentation을 위한 열거형\n',
    'ConfirmMeasure': '\n논문에 제시된 direct confirm measure를 위한 열거형\n',
    'IndirectMeasure': '\n논문에 제시된 indirect confirm measure를 위한 열거형\n',

    # Corpus

    # C++ docstrings (from docs.h and label_docs.h)
    'CTModel': '''.. versionadded:: 0.2.0
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
''',
    'CTModel.alpha': '''이 프로퍼티는 `CTModel`에서 사용불가합니다. 대신 `CTModel.prior_mean`와 `CTModel.prior_cov`를 사용하십시오.

.. versionadded:: 0.9.1''',
    'CTModel.get_correlations': '''토픽 `topic_id`와 나머지 토픽들 간의 상관관계를 반환합니다.
반환값은 `tomotopy.models.LDAModel.k` 길이의 `float`의 `list`입니다.

Parameters
----------
topic_id : Union[int, None]
    토픽을 지정하는 [0, `k`), 범위의 정수

    생략 시 상관계수 행렬 전체가 반환됩니다.
''',
    'CTModel.num_beta_sample': '''beta 파라미터를 표집하는 횟수, 기본값은 10.

CTModel은 각 문헌마다 총 `num_beta_sample` 개수의 beta 파라미터를 표집합니다.
beta 파라미터를 더 많이 표집할 수록, 전체 분포는 정교해지지만 학습 시간이 더 많이 걸립니다.
만약 모형 내에 문헌의 개수가 적은 경우 이 값을 크게하면 더 정확한 결과를 얻을 수 있습니다.
''',
    'CTModel.num_tmn_sample': '''절단된 다변수 정규분포에서 표본을 추출하기 위한 반복 횟수, 기본값은 5.

만약 결과에서 토픽 간 상관관계가 편향되게 나올 경우 이 값을 키우면 편향을 해소하는 데에 도움이 될 수 있습니다.
''',
    'CTModel.prior_cov': '토픽의 사전 분포인 로지스틱 정규 분포의 공분산 행렬 (읽기전용)',
    'CTModel.prior_mean': '토픽의 사전 분포인 로지스틱 정규 분포의 평균 벡터 (읽기전용)',
    'Candidate.cf': '후보의 장서빈도(읽기전용)',
    'Candidate.df': '후보의 문헌빈도(읽기전용)',
    'Candidate.name': '토픽 레이블의 실제 이름',
    'Candidate.score': '후보의 점수(읽기전용)',
    'Candidate.words': '토픽 레이블의 후보 (읽기전용)',
    'DMRModel': '''이 타입은 Dirichlet Multinomial Regression(DMR) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'DMRModel.add_doc': '''현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

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
''',
    'DMRModel.alpha': '''각 메타데이터별 문헌-토픽 분포의 사전 분포, `[k, f]` 모양. `np.exp(DMRModel.lambdas)`와 동일 (읽기전용)

.. versionadded:: 0.9.0

.. warning::

    0.11.0 버전 전까지는 lambda getter에 있는 버그로 잘못된 값이 출력되었습니다. 0.11.0 이후 버전으로 업그레이드하시길 권장합니다.''',
    'DMRModel.alpha_epsilon': '평탄화 계수 alpha-epsilon (읽기전용)',
    'DMRModel.f': '메타데이터 자질 종류의 개수 (읽기전용)',
    'DMRModel.get_topic_prior': '''.. versionadded:: 0.12.0

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
''',
    'DMRModel.lambda_': '''현재 모형의 lambda 파라미터을 보여주는 `[k, len(metadata_dict), l]` 모양의 float array (읽기전용)

lambda 파라미터와 토픽 사전 분포 간의 관계에 대해서는 `tomotopy.models.DMRModel.get_topic_prior`를 참고하십시오.

.. versionadded:: 0.12.0''',
    'DMRModel.lambdas': '''현재 모형의 lambda 파라미터을 보여주는 `[k, f]` 모양의 float array (읽기전용)

.. warning::

    0.11.0 버전 전까지는 lambda getter에 있는 버그로 잘못된 값이 출력되었습니다. 0.11.0 이후 버전으로 업그레이드하시길 권장합니다.''',
    'DMRModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

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
''',
    'DMRModel.metadata_dict': '`tomotopy.Dictionary` 타입의 메타데이터 사전 (읽기전용)',
    'DMRModel.multi_metadata_dict': '''`tomotopy.Dictionary` 타입의 메타데이터 사전 (읽기전용)

.. versionadded:: 0.12.0

    이 사전은 `metadata_dict`와는 별개입니다.
''',
    'DMRModel.sigma': '하이퍼 파라미터 sigma (읽기전용)',
    'DTModel': '''이 타입은 Dynamic Topic Model의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'DTModel.add_doc': '''현재 모델에 `timepoint` 시점의 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
''',
    'DTModel.alpha': '''문헌별 토픽 분포, `[num_timepoints, k]` 모양 (읽기전용)

.. versionadded:: 0.9.0''',
    'DTModel.eta': '''이 프로퍼티는 `DTModel`에서 사용불가합니다. 대신 `DTModel.docs[x].eta`를 사용하십시오.

.. versionadded:: 0.9.0''',
    'DTModel.get_alpha': '''`timepoint` 시점에 대한 alpha 파라미터의 리스트를 반환합니다.

Parameters
----------
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
''',
    'DTModel.get_count_by_topics': '''각각의 시점과 토픽에 할당된 단어의 개수를 `[num_timepoints, k]` 모양으로 반환합니다.

.. versionadded:: 0.9.0''',
    'DTModel.get_phi': '''`timepoint` 시점의 `topic_id`에 대한 phi 파라미터의 리스트를 반환합니다.

Parameters
----------
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
topic_id : int
    토픽을 나타내는 [0, `k`) 범위의 정수
''',
    'DTModel.get_topic_word_dist': '''시점 `timepoint`의 토픽 `topic_id`의 단어 분포를 반환합니다.
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
''',
    'DTModel.get_topic_words': '''시점 `timepoint`의 토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
timepoint : int
	시점을 가리키는 [0, `t`) 범위의 정수
''',
    'DTModel.lr_a': 'SGLD의 스텝 크기를 결정하는 0보다 큰 파라미터 `a` (e_i = a * (b + i) ^ -c)',
    'DTModel.lr_b': 'SGLD의 스텝 크기를 결정하는 0 이상의 파라미터 `b` (e_i = a * (b + i) ^ -c)',
    'DTModel.lr_c': 'SGLD의 스텝 크기를 결정하는 (0.5, 1] 범위의 파라미터 `c` (e_i = a * (b + i) ^ -c)',
    'DTModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
timepoint : int
    시점을 나타내는 [0, `t`) 범위의 정수
''',
    'DTModel.num_docs_by_timepoint': '각 시점별 모델 내 문헌 개수 (읽기전용)',
    'DTModel.num_timepoints': '모델의 시점 개수 (읽기전용)',
    'Document': '이 타입은 토픽 모델에 사용되는 문헌들에 접근할 수 있는 추상 인터페이스을 제공합니다.',
    'Document.beta': '''문헌의 각 토픽별 beta 파라미터를 보여주는 `list` (`tomotopy.models.CTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.2.0''',
    'Document.eta': '''문헌의 eta 파라미터(토픽 분포)를 나타내는 `list` (`tomotopy.models.DTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.0''',
    'Document.get_count_vector': '''.. versionadded:: 0.7.0

현재 문헌의 카운트 벡터를 반환합니다.''',
    'Document.get_ll': '''.. versionadded:: 0.10.0

현재 문헌의 로그가능도 총합을 반환합니다.''',
    'Document.get_sub_topic_dist': '''.. versionadded:: 0.5.0

현재 문헌의 하위 토픽 확률 분포를 `list` 형태로 반환합니다. (`tomotopy.models.PAModel` 전용)

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
''',
    'Document.get_sub_topics': '''.. versionadded:: 0.5.0

현재 문헌의 상위 `top_n`개의 하위 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다. (`tomotopy.models.PAModel` 전용)''',
    'Document.get_topic_dist': '''현재 문헌의 토픽 확률 분포를 `list` 형태로 반환합니다.

Parameters
----------
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    참일 경우 가상 문헌의 토픽 분포를 반환합니다. `tomotopy.models.PTModel`에서만 유효합니다.
''',
    'Document.get_topics': '''현재 문헌의 상위 `top_n`개의 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다.
    
Parameters
----------
top_n : int
    "상위-n"에서 n의 값
from_pseudo_doc : bool
    .. versionadded:: 0.12.2

    참일 경우 가상 문헌의 토픽 분포를 반환합니다. `tomotopy.models.PTModel`에서만 유효합니다.
''',
    'Document.get_words': '''.. versionadded:: 0.4.2

현재 문헌의 상위 `top_n`개의 단어와 그 확률을 `tuple`의 `list` 형태로 반환합니다.''',
    'Document.labels': '''문헌에 매겨진 (레이블, 레이블에 속하는 각 주제의 확률들)의 `list` (`tomotopy.models.LLDAModel`, `tomotopy.models.PLDAModel` 모형에서만 사용됨 , 읽기전용)

.. versionadded:: 0.3.0''',
    'Document.metadata': '문헌의 범주형 메타데이터 (`tomotopy.models.DMRModel`과 `tomotopy.models.GDMRModel` 모형에서만 사용됨, 읽기전용)',
    'Document.multi_metadata': '''문헌의 범주형 메타데이터 (`tomotopy.models.DMRModel`과 `tomotopy.models.GDMRModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.12.0''',
    'Document.numeric_metadata': '''문헌의 연속형 숫자 메타데이터 (`tomotopy.models.GDMRModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.11.0''',
    'Document.paths': '''주어진 문헌에 대한 깊이별 토픽 번호의 `list` (`tomotopy.models.HLDAModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.1''',
    'Document.pseudo_doc_id': '''문헌이 할당된 가상 문헌의 id (`tomotopy.models.PTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.11.0''',
    'Document.raw': '문헌의 가공되지 않는 전체 텍스트 (읽기전용)',
    'Document.span': '문헌의 각 단어 토큰의 구간(바이트 단위 시작 지점과 끝 지점의 tuple) (읽기전용)',
    'Document.subtopics': '문헌의 단어들이 각각 할당된 하위 토픽을 보여주는 `list` (`tomotopy.models.PAModel`와 `tomotopy.models.HPAModel` 모형에서만 사용됨, 읽기전용)',
    'Document.timepoint': '''문헌의 시점 (`tomotopy.models.DTModel` 모형에서만 사용됨, 읽기전용)

.. versionadded:: 0.7.0''',
    'Document.topics': '''문헌의 단어들이 각각 할당된 토픽을 보여주는 `list` (읽기 전용)

`tomotopy.models.PAModel`와 `tomotopy.models.HPAModel` 모형에서는 이 값이 상위토픽의 ID를 가리킵니다.''',
    'Document.uid': '문헌의 고유 ID (읽기전용)',
    'Document.vars': '''문헌의 응답 변수를 보여주는 `list` (`tomotopy.models.SLDAModel` 모형에서만 사용됨 , 읽기전용)

.. versionadded:: 0.2.0''',
    'Document.weights': '문헌의 가중치 (읽기전용)',
    'Document.windows': '문헌의 단어들이 할당된 윈도우의 ID를 보여주는 `list` (`tomotopy.models.MGLDAModel` 모형에서만 사용됨, 읽기전용)',
    'Document.words': '문헌 내 단어들의 ID가 담긴 `list` (읽기전용)',
    'FoRelevance': '''.. versionadded:: 0.6.0

First-order Relevance를 이용한 토픽 라벨링 기법을 제공합니다. 이 구현체는 다음 논문에 기초하고 있습니다:

> * Mei, Q., Shen, X., & Zhai, C. (2007, August). Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499).

Parameters
----------
topic_model
    토픽명을 붙일 토픽 모델의 인스턴스
cands : Iterable[tomotopy.label.Candidate]
    토픽명으로 사용될 후보들의 리스트
min_df : int
    사용하려는 후보의 최소 문헌 빈도. 연어가 등장하는 문헌 수가 `min_df`보다 작은 경우 선택에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
smoothing : float
    라플라스 평활화에 사용될 0보다 큰 작은 실수
mu : float
    변별성 계수. 이 계수가 클 때, 특정 토픽에 대해서만 높은 점수를 가지고 나머지 토픽에 대해서는 낮은 점수를 가진 후보가 더 높은 최종 점수를 받습니다.
window_size : int
    .. versionadded:: 0.10.0
    
    동시출현 빈도를 계산하기 위한 슬라이딩 윈도우의 크기. -1로 설정시 슬라이딩 윈도우를 사용하지 않고, 문헌 전체를 활용해 동시출현 빈도를 계산합니다.
    분석에 사용하는 문헌들의 길이가 길다면, 이 값을 -1이 아닌 50 ~ 100 정도로 설정하는 걸 권장합니다.
workers : int
    깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
    만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
''',
    'FoRelevance.get_topic_labels': '''토픽 `k`에 해당하는 레이블 후보 상위 n개를 반환합니다.

Parameters
----------
k : int
    토픽을 지정하는 정수
top_n : int
    토픽 레이블의 개수''',
    'GDMRModel': '''이 타입은 Generalized DMR(g-DMR) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Lee, M., & Song, M. Incorporating citation impact into analysis of research trends. Scientometrics, 1-34.

.. versionadded:: 0.8.0

.. warning::

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.models.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
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
''',
    'GDMRModel.add_doc': '''현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

.. versionchanged:: 0.11.0

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.models.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
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
''',
    'GDMRModel.decay': '하이퍼 파라미터 decay (읽기전용)',
    'GDMRModel.degrees': '르장드르 다항식의 차수 (읽기전용)',
    'GDMRModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

.. versionchanged:: 0.11.0

    0.10.2버전까지는 `metadata`가 숫자형 연속 변수를 표현하는데 사용되었고, 별도로 범주형 변수에 사용되는 인자가 없었습니다.
    0.11.0버전부터는 `tomotopy.models.DMRModel`과의 통일성을 위해 기존의 `metadata` 인수가 `numeric_metadata`라는 이름으로 변경되고,
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
''',
    'GDMRModel.metadata_range': '각 메타데이터 변수의 범위를 나타내는 `list` (읽기전용)',
    'GDMRModel.sigma0': '하이퍼 파라미터 sigma0 (읽기전용)',
    'GDMRModel.tdf': '''주어진 `metadata`에 대해 토픽 분포를 계산하여, `k` 길이의 list로 반환합니다.

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
''',
    'GDMRModel.tdf_linspace': '''주어진 `metadata`에 대해 토픽 분포를 계산하여, `k` 길이의 list로 반환합니다.

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
''',
    'HDPModel': '''이 타입은 Hierarchical Dirichlet Process(HDP) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'HDPModel.convert_to_lda': '''.. versionadded:: 0.8.0

현재의 HDP 모델을 동등한 LDA모델로 변환하고, `(new_lda_mode, new_topic_id)`를 반환합니다.
이 때 `topic_threshold`보다 작은 비율의 토픽은 `new_lda_model`에서 제거됩니다.

`new_topic_id`는 길이 `HDPModel.k`의 배열이며, `new_topic_id[i]`는 새 LDA 모델에서 원 HDP 모델의 토픽 `i`와 동등한 토픽의 id를 가리킵니다.
만약 원 HDP 모델의 토픽 `i`가 유효하지 않거나, 새 LDA 모델에서 제거된 것이라면, `new_topic_id[i]`는 `-1`이 됩니다.

Parameters
----------
topic_threshold : float
    이 값보다 작은 비율의 토픽은 새 LDA 모델에서 제거됩니다.
    기본값은 0이며, 이 경우 유효하지 않는 토픽을 제외한 모든 토픽이 LDA 모델에 포함됩니다.
''',
    'HDPModel.gamma': '하이퍼 파라미터 gamma (읽기전용)',
    'HDPModel.is_live_topic': '''`topic_id`가 유효한 토픽을 가리키는 경우 `True`, 아닌 경우 `False`를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HDPModel.live_k': '현재 모델 내의 유효한 토픽의 개수 (읽기전용)',
    'HDPModel.num_tables': '현재 모델 내의 총 테이블 개수 (읽기전용)',
    'HDPModel.purge_dead_topics': '''.. versionadded:: 0.12.3

현재 모델에서 유효하지 않은 토픽을 모두 제거하고 `new_topic_ids`를 반환합니다. 호출 후에 `HDPModel.k`는 `HDPModel.live_k`값으로 줄어들며 모든 토픽은 유효한 상태가 됩니다.

`new_topic_id`는 길이 `HDPModel.k`의 배열이며, `new_topic_id[i]`는 새 모델에서 기존 HDP 모델의 토픽 `i`와 동등한 토픽의 id를 가리킵니다.
만약 기존 HDP 모델의 토픽 `i`가 유효하지 않거나, 새 모델에서 제거된 것이라면, `new_topic_id[i]`는 `-1`이 됩니다.
''',
    'HLDAModel': '''이 타입은 Hierarchical LDA 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'HLDAModel.children_topics': '''`topic_id` 토픽의 자식 토픽들의 ID를 list로 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HLDAModel.depth': '현재 모델의 총 깊이 (읽기전용)',
    'HLDAModel.gamma': '하이퍼 파라미터 gamma (읽기전용)',
    'HLDAModel.is_live_topic': '''`topic_id`가 유효한 토픽을 가리키는 경우 `True`, 아닌 경우 `False`를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HLDAModel.level': '''`topic_id` 토픽의 레벨을 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HLDAModel.live_k': '현재 모델 내의 유효한 토픽의 개수 (읽기전용)',
    'HLDAModel.num_docs_of_topic': '''`topic_id` 토픽에 속하는 문헌의 개수를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HLDAModel.parent_topic': '''`topic_id` 토픽의 부모 토픽의 ID를 반환합니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
''',
    'HPAModel': '''이 타입은 Hierarchical Pachinko Allocation(HPA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'HPAModel.alpha': '''문헌의 상위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1 + 1]` 모양.
0번째 요소는 최상위 토픽을 가리키며, 1 ~ k1번째가 상위 토픽을 가리킨다. (읽기전용)

.. versionadded:: 0.9.0''',
    'HPAModel.get_topic_word_dist': '''토픽 `topic_id`의 단어 분포를 반환합니다.
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
''',
    'HPAModel.get_topic_words': '''토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    0일 경우 최상위 토픽을 가리키며,
    [1, 1 + `k1`) 범위의 정수는 상위 토픽을,
    [1 + `k1`, 1 + `k1` + `k2`) 범위의 정수는 하위 토픽을 가리킵니다.
''',
    'HPAModel.subalpha': '''상위 토픽의 하위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1, k2 + 1]` 모양.
`[x, 0]` 요소는 상위 토픽 `x`를 가리키며, `[x, 1 ~ k2]` 요소는 상위 토픽 `x` 내의 하위 토픽들을 가리킨다. (읽기전용)

.. versionadded:: 0.9.0''',
    'LDAModel': '''이 타입은 Latent Dirichlet Allocation(LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
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
''',
    'LDAModel.add_corpus': '''.. versionadded:: 0.10.0

코퍼스를 이용해 현재 모델에 새로운 문헌들을 추가하고 추가된 문헌로 구성된 새 코퍼스를 반환합니다. 
이 메소드는 `tomotopy.models.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.
Parameters
----------
corpus : tomotopy.utils.Corpus
    토픽 모델에 추가될 문헌들로 구성된 코퍼스
transform : Callable[dict, dict]
    특정한 토픽 모델에 맞춰 임의 키워드 인자를 조작하기 위한 호출가능한 객체
''',
    'LDAModel.add_doc': '''현재 모델에 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다. 이 메소드는 `tomotopy.models.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable''',
    'LDAModel.alpha': '문헌의 토픽 분포에 대한 디리클레 분포 파라미터 (읽기전용)',
    'LDAModel.burn_in': '''파라미터 학습 초기의 Burn-in 단계의 반복 횟수를 얻거나 설정합니다.

기본값은 0입니다.''',
    'LDAModel.copy': '''.. versionadded:: 0.12.0

깊게 복사된 새 인스턴스를 반환합니다.''',
    'LDAModel.docs': '현재 모델에 포함된 `tomotopy.utils.Document`에 접근할 수 있는 `list`형 인터페이스 (읽기전용)',
    'LDAModel.eta': '하이퍼 파라미터 eta (읽기전용)',
    'LDAModel.get_count_by_topics': '각각의 토픽에 할당된 단어의 개수를 `list`형태로 반환합니다.',
    'LDAModel.get_topic_word_dist': '''토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
''',
    'LDAModel.get_topic_words': '''토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    토픽을 가리키는 [0, `k`) 범위의 정수
top_n : int
	반환할 단어의 개수
return_id : bool
	참일 경우 단어 ID도 함께 반환합니다.
''',
    'LDAModel.get_word_prior': '''.. versionadded:: 0.6.0

`word`에 대한 사전 주제 분포를 반환합니다. 별도로 설정된 값이 없을 경우 빈 리스트가 반환됩니다.

Parameters
----------
word : str
    어휘
''',
    'LDAModel.global_step': '''현재까지 수행된 학습의 총 반복 횟수 (읽기전용)

.. versionadded:: 0.9.0''',
    'LDAModel.infer': '''새로운 문헌인 `doc`에 대해 각각의 주제 분포를 추론하여 반환합니다.
반환 타입은 (`doc`의 주제 분포, 로그가능도) 또는 (`doc`의 주제 분포로 구성된 `list`, 로그가능도)입니다.

Parameters
----------
doc : Union[tomotopy.utils.Document, Iterable[tomotopy.utils.Document], tomotopy.utils.Corpus]
    추론에 사용할 `tomotopy.utils.Document`의 인스턴스이거나 이 인스턴스들의 `list`.
    이 인스턴스들은 `tomotopy.models.LDAModel.make_doc` 메소드를 통해 얻을 수 있습니다.

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
    `doc`이 `tomotopy.utils.Document`로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`입니다.
    
    `doc`이 `tomotopy.utils.Document`의 list로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`의 list입니다.
    
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우, `result`는 추론된 결과 문서들을 담고 있는, `tomotopy.utils.Corpus`의 새로운 인스턴스입니다.
    각 문헌별 토픽 분포를 얻기 위해서는 `tomotopy.utils.Document.get_topic_dist`를 사용하면 됩니다.
log_ll : float
    각 문헌별 로그 가능도의 리스트
''',
    'LDAModel.k': '토픽의 개수 (읽기전용)',
    'LDAModel.ll_per_word': '현재 모델의 단어당 로그 가능도 (읽기전용)',
    'LDAModel.load': '`filename` 경로의 파일로부터 모델 인스턴스를 읽어들여 반환합니다.',
    'LDAModel.loads': 'bytes-like object인 `data`로로부터 모델 인스턴스를 읽어들여 반환합니다.',
    'LDAModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다..

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
''',
    'LDAModel.num_vocabs': '''작은 빈도의 단어들을 제거한 뒤 남은 어휘의 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.

.. deprecated:: 0.8.0

    이 프로퍼티의 이름은 혼동을 일으킬 여지가 있어 제거될 예정입니다. 대신 `len(used_vocabs)`을 사용하십시오.''',
    'LDAModel.num_words': '''현재 모델에 포함된 문헌들 전체의 단어 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.''',
    'LDAModel.optim_interval': '''파라미터 최적화의 주기를 얻거나 설정합니다.

기본값은 10이며, 0으로 설정할 경우 학습 과정에서 파라미터 최적화를 수행하지 않습니다.''',
    'LDAModel.perplexity': '현재 모델의 Perplexity (읽기전용)',
    'LDAModel.removed_top_words': '모델 생성시 `rm_top` 파라미터를 1 이상으로 설정한 경우, 빈도수가 높아서 모델에서 제외된 단어의 목록을 보여줍니다. (읽기전용)',
    'LDAModel.save': '''현재 모델을 `filename` 경로의 파일에 저장합니다. `None`을 반환합니다.

`full`이 `True`일 경우, 모델의 전체 상태가 파일에 모두 저장됩니다. 저장된 모델을 다시 읽어들여 학습(`train`)을 더 진행하고자 한다면 `full` = `True`로 하여 저장하십시오.
반면 `False`일 경우, 토픽 추론에 관련된 파라미터만 파일에 저장됩니다. 이 경우 파일의 용량은 작아지지만, 추가 학습은 불가하고 새로운 문헌에 대해 추론(`infer`)하는 것만 가능합니다.

.. versionadded:: 0.6.0

0.6.0 버전부터 모델 파일 포맷이 변경되었습니다.
따라서 0.6.0 이후 버전에서 저장된 모델 파일 포맷은 0.5.2 버전 이전과는 호환되지 않습니다.
''',
    'LDAModel.saves': '''.. versionadded:: 0.11.0

현재 모델을 직렬화하여 `bytes`로 만든 뒤 이를 반환합니다. 인자는 `tomotopy.models.LDAModel.save`와 동일하게 작동합니다.''',
    'LDAModel.set_word_prior': '''.. versionadded:: 0.6.0

어휘-주제 사전 분포를 설정합니다. 이 메소드는 `tomotopy.models.LDAModel.train`를 호출하기 전에만 사용될 수 있습니다.

Parameters
----------
word : str
    설정할 어휘
prior : Union[Iterable[float], Dict[int, float]]
    어휘 `word`의 주제 분포. `prior`의 길이는 `tomotopy.models.LDAModel.k`와 동일해야 합니다.

Note
----
0.12.6 버전부터 이 메소드는 `prior`에 리스트 타입 파라미터 외에도 딕셔너리 타입 파라미터를 받을 수 있습니다.
딕셔너리의 키는 주제의 id이며 값은 사전 주제 분포입니다. 만약 주제의 사전 분포가 설정되지 않았을 경우, 기본값으로 모델의 `eta` 파라미터로 설정됩니다.
```python
>>> model = tp.LDAModel(k=3, eta=0.01)
>>> model.set_word_prior(\'apple\', [0.01, 0.9, 0.01])
>>> model.set_word_prior(\'apple\', {1: 0.9}) # 위와 동일한 효과
```
''',
    'LDAModel.summary': '''.. versionadded:: 0.9.0

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
''',
    'LDAModel.train': '''깁스 샘플링을 `iter` 회 반복하여 현재 모델을 학습시킵니다. 반환값은 `None`입니다. 
이 메소드가 호출된 이후에는 더 이상 `tomotopy.models.LDAModel.add_doc`로 현재 모델에 새로운 학습 문헌을 추가시킬 수 없습니다.

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

    학습 시 새로운 토픽을 생성하지 못하도록 합니다. 이 파라미터는 오직 `tomotopy.models.HLDAModel`에만 유효합니다.
callback_interval : int
    .. versionadded:: 0.12.6

    `callback` 함수를 호출하는 간격. `callback_interval` <= 0일 경우 학습 시작과 종료 시에만 `callback` 함수가 호출됩니다.
callback : Callable[[tomotopy.models.LDAModel, int, int], None]
    .. versionadded:: 0.12.6

    학습 과정에서 `callback_interval` 마다 호출되는 호출가능한 객체. 
    이 함수는 세 개의 인자를 받습니다: 현재 모델, 현재까지의 반복 횟수, 총 반복 횟수.
show_progress : bool
    .. versionadded:: 0.12.6

    `True`일 경우 `tqdm` 패키지를 이용해 학습 진행 상황을 표시합니다.
''',
    'LDAModel.tw': '현재 모델의 용어 가중치 계획 (읽기전용)',
    'LDAModel.used_vocab_df': '''모델에 실제로 사용된 어휘들의 문헌빈도를 보여주는 `list` (읽기전용)

.. versionadded:: 0.8.0''',
    'LDAModel.used_vocab_freq': '모델에 실제로 사용된 어휘들의 빈도를 보여주는 `list` (읽기전용)',
    'LDAModel.used_vocab_weighted_freq': '모델에 실제로 사용된 어휘들의 빈도(용어 가중치 적용됨)를 보여주는 `list` (읽기전용)',
    'LDAModel.used_vocabs': '''모델에 실제로 사용된 어휘만을 포함하는 `tomotopy.Dictionary` 타입의 어휘 사전 (읽기전용)

.. versionadded:: 0.8.0''',
    'LDAModel.vocab_df': '''빈도수로 필터링된 어휘와 현재 모델에 포함된 어휘 전체의 문헌빈도를 보여주는 `list` (읽기전용)

.. versionadded:: 0.8.0''',
    'LDAModel.vocab_freq': '빈도수로 필터링된 어휘와 현재 모델에 포함된 어휘 전체의 빈도를 보여주는 `list` (읽기전용)',
    'LDAModel.vocabs': '빈도수로 필터링된 어휘와 모델에 포함된 어휘 전체를 포함하는 `tomotopy.Dictionary` 타입의 어휘 사전 (읽기전용)',
    'LLDAModel': '''이 타입은 Labeled LDA(L-LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.

.. versionadded:: 0.3.0

.. deprecated:: 0.11.0
    `tomotopy.models.PLDAModel`를 대신 사용하세요.

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
''',
    'LLDAModel.add_doc': '''현재 모델에 `labels`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
labels : Iterable[str]
    문헌의 레이블 리스트
''',
    'LLDAModel.get_topic_words': '''토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    전체 레이블의 개수를 `l`이라고 할 때, [0, `l`) 범위의 정수는 각각의 레이블에 해당하는 토픽을 가리킵니다. 
    해당 토픽의 레이블 이름은 `tomotopy.models.LLDAModel.topic_label_dict`을 열람하여 확인할 수 있습니다.
    [`l`, `k`) 범위의 정수는 어느 레이블에도 속하지 않는 잠재 토픽을 가리킵니다.
''',
    'LLDAModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
labels : Iterable[str]
    문헌의 레이블 리스트
''',
    'LLDAModel.topic_label_dict': '`tomotopy.Dictionary` 타입의 토픽 레이블 사전 (읽기전용)',
    'MGLDAModel': '''이 타입은 Multi Grain Latent Dirichlet Allocation(MG-LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'MGLDAModel.add_doc': '''현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
delimiter : str
    문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
''',
    'MGLDAModel.alpha_g': '하이퍼 파라미터 alpha_g (읽기전용)',
    'MGLDAModel.alpha_l': '하이퍼 파라미터 alpha_l (읽기전용)',
    'MGLDAModel.alpha_mg': '하이퍼 파라미터 alpha_mg (읽기전용)',
    'MGLDAModel.alpha_ml': '하이퍼 파라미터 alpha_ml (읽기전용)',
    'MGLDAModel.eta_g': '하이퍼 파라미터 eta_g (읽기전용)',
    'MGLDAModel.eta_l': '하이퍼 파라미터 eta_l (읽기전용)',
    'MGLDAModel.gamma': '하이퍼 파라미터 gamma (읽기전용)',
    'MGLDAModel.get_topic_word_dist': '''토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
topic_id : int
    [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
''',
    'MGLDAModel.get_topic_words': '''토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
''',
    'MGLDAModel.k_g': '하이퍼 파라미터 k_g (읽기전용)',
    'MGLDAModel.k_l': '하이퍼 파라미터 k_l (읽기전용)',
    'MGLDAModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
delimiter : str
    문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
''',
    'MGLDAModel.t': '하이퍼 파라미터 t (읽기전용)',
    'PAModel': '''이 타입은 Pachinko Allocation(PA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

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
''',
    'PAModel.alpha': '''문헌의 상위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1]` 모양 (읽기전용)

.. versionadded:: 0.9.0''',
    'PAModel.get_count_by_super_topic': '''각각의 상위 토픽에 할당된 단어의 개수를 `list`형태로 반환합니다.

.. versionadded:: 0.9.0''',
    'PAModel.get_sub_topic_dist': '''상위 토픽 `super_topic_id`의 하위 토픽 분포를 반환합니다.
반환하는 값은 현재 상위 토픽 내 각각의 하위 토픽들의 발생확률을 나타내는 `k2`개의 소수로 구성된 `list`입니다.

Parameters
----------
super_topic_id : int
    상위 토픽을 가리키는 [0, `k1`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
''',
    'PAModel.get_sub_topics': '''.. versionadded:: 0.1.4

상위 토픽 `super_topic_id`에 속하는 상위 `top_n`개의 하위 토픽과 각각의 확률을 반환합니다. 
반환 타입은 (하위토픽:`int`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
super_topic_id : int
    상위 토픽을 가리키는 [0, `k1`) 범위의 정수
''',
    'PAModel.get_topic_word_dist': '''하위 토픽 `sub_topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 하위 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

Parameters
----------
sub_topic_id : int
    하위 토픽을 가리키는 [0, `k2`) 범위의 정수
normalize : bool
    .. versionadded:: 0.11.0

    참일 경우 총합이 1이 되는 확률 분포를 반환하고, 거짓일 경우 정규화되지 않는 값을 그대로 반환합니다.
''',
    'PAModel.get_topic_words': '''하위 토픽 `sub_topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
sub_topic_id : int
    하위 토픽을 가리키는 [0, `k2`) 범위의 정수
''',
    'PAModel.infer': '''.. versionadded:: 0.5.0

새로운 문헌인 `doc`에 대해 각각의 주제 분포를 추론하여 반환합니다.
반환 타입은 ((`doc`의 주제 분포, `doc`의 하위 주제 분포), 로그가능도) 또는 ((`doc`의 주제 분포, `doc`의 하위 주제 분포)로 구성된 `list`, 로그가능도)입니다.

Parameters
----------
doc : Union[tomotopy.utils.Document, Iterable[tomotopy.utils.Document]]
    추론에 사용할 `tomotopy.utils.Document`의 인스턴스이거나 이 인스턴스들의 `list`.
    이 인스턴스들은 `tomotopy.models.LDAModel.make_doc` 메소드를 통해 얻을 수 있습니다.

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
    `doc`이 `tomotopy.utils.Document`로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`와 하위 토픽 분포를 나타내는 `List[float]`의 tuple입니다.
    
    `doc`이 `tomotopy.utils.Document`의 list로 주어진 경우, `result`는 문헌의 토픽 분포를 나타내는 `List[float]`와 하위 토픽 분포를 나타내는 `List[float]`의 tuple의 list입니다.
    
    `doc`이 `tomotopy.utils.Corpus`의 인스턴스로 주어진 경우, `result`는 추론된 결과 문서들을 담고 있는, `tomotopy.utils.Corpus`의 새로운 인스턴스입니다.
    각 문헌별 토픽 분포를 얻기 위해서는 `tomotopy.utils.Document.get_topic_dist`, 하위 토픽 분포를 얻기 위해서는 `tomotopy.utils.Document.get_sub_topic_dist`를 사용하면 됩니다.
log_ll : List[float]
    각 문헌별 로그 가능도의 리스트
''',
    'PAModel.k1': 'k1, 상위 토픽의 개수 (읽기전용)',
    'PAModel.k2': 'k2, 하위 토픽의 개수 (읽기전용)',
    'PAModel.subalpha': '''상위 토픽의 하위 토픽 분포에 대한 디리클레 분포 파라미터, `[k1, k2]` 모양 (읽기전용)

.. versionadded:: 0.9.0''',
    'PLDAModel': '''이 타입은 Partially Labeled LDA(PLDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
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
''',
    'PLDAModel.get_topic_words': '''토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

Parameters
----------
topic_id : int
    전체 레이블의 개수를 `l`이라고 할 때, [0, `l` * `topics_per_label`) 범위의 정수는 각각의 레이블에 해당하는 토픽을 가리킵니다. 
    해당 토픽의 레이블 이름은 `tomotopy.models.PLDAModel.topic_label_dict`을 열람하여 확인할 수 있습니다.
    [`l` * `topics_per_label`, `l` * `topics_per_label` + `latent_topics`) 범위의 정수는 어느 레이블에도 속하지 않는 잠재 토픽을 가리킵니다.
''',
    'PLDAModel.latent_topics': '잠재 토픽의 개수 (읽기전용)',
    'PLDAModel.topic_label_dict': '`tomotopy.Dictionary` 타입의 토픽 레이블 사전 (읽기전용)',
    'PLDAModel.topics_per_label': '레이블별 토픽의 개수 (읽기전용)',
    'PMIExtractor': '''.. versionadded:: 0.6.0

`PMIExtractor`는 다변수 점별 상호정보량을 활용해 연어를 추출합니다. 이는 통계적으로 자주 함께 등장하는 단어열을 찾아줍니다.

Parameters
----------
min_cf : int
    추출하려는 후보의 최소 장서 빈도. 문헌 내 등장하는 빈도수가 `min_cf`보다 작은 연어는 후보에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
min_df : int
    추출하려는 후보의 최소 문헌 빈도. 연어가 등장하는 문헌 수가 `min_df`보다 작은 경우 후보에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
min_len : int
    .. versionadded:: 0.10.0
    
    분석하려는 연어의 최소 길이. 1로 설정시 단일 단어들도 모두 추출합니다.
    단일 단어들은 `max_cand` 개수 계산에서 제외됩니다.
max_len : int
    분석하려는 연어의 최대 길이
max_cand : int
    추출하려는 후보의 최대 개수
''',
    'PMIExtractor.extract': '''`topic_model`로부터 추출된 토픽 레이블 후보인 `tomotopy.label.Candidate`의 리스트를 반환합니다.

Parameters
----------
topic_model
    후보를 추출할 문헌들을 가지고 있는 토픽 모델의 인스턴스
''',
    'PTModel': '''.. versionadded:: 0.11.0
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
''',
    'PTModel.p': '''가상 문헌의 개수 (읽기전용)

.. versionadded:: 0.11.0''',
    'SLDAModel': '''이 타입은 supervised Latent Dirichlet Allocation(sLDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
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
    
    > * \'l\': 선형 변수 (아무 실수 값이나 가능)
    > * \'b\': 이진 변수 (0 혹은 1만 가능)
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
''',
    'SLDAModel.add_doc': '''현재 모델에 응답 변수 `y`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
y : Iterable[float]
    문헌의 응답 변수로 쓰일 `float`의 `list`. `y`의 길이는 모델의 응답 변수의 개수인 `tomotopy.models.SLDAModel.f`와 일치해야 합니다.
    
    .. versionchanged:: 0.5.1
    
        만약 결측값이 있을 경우, 해당 항목을 `NaN`으로 설정할 수 있습니다. 이 경우 `NaN`값을 가진 문헌은 토픽을 모델링하는 데에는 포함되지만, 응답 변수 회귀에서는 제외됩니다.
''',
    'SLDAModel.estimate': '''`doc`의 추정된 응답 변수를 반환합니다.
만약 `doc`이 `tomotopy.models.SLDAModel.make_doc`에 의해 생성된 인스턴스라면, 먼저 `tomotopy.models.LDAModel.infer`를 통해 토픽 추론을 실시한 다음 이 메소드를 사용해야 합니다.

Parameters
----------
doc : tomotopy.utils.Document
    응답 변수를 추정하려하는 문헌의 인스턴스 혹은 인스턴스들의 list
''',
    'SLDAModel.f': '응답 변수의 개수 (읽기전용)',
    'SLDAModel.get_regression_coef': '''응답 변수 `var_id`의 회귀 계수를 반환합니다.

Parameters
----------
var_id : int
    응답 변수를 지정하는 [0, `f`) 범위의 정수

    생략시, `[f, k]` 모양의 전체 회귀 계수가 반환됩니다.
''',
    'SLDAModel.get_var_type': '응답 변수 `var_id`의 종류를 반환합니다. \'l\'은 선형 변수, \'b\'는 이진 변수를 뜻합니다.',
    'SLDAModel.make_doc': '''`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.utils.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.models.LDAModel.infer` 메소드에 사용될 수 있습니다.

Parameters
----------
words : Iterable[str]
    문헌의 각 단어를 나열하는 `str` 타입의 iterable
y : Iterable[float]
    문헌의 응답 변수로 쓰일 `float`의 `list`. 
    `y`의 길이는 모델의 응답 변수의 개수인 `tomotopy.models.SLDAModel.f`와 꼭 일치할 필요는 없습니다.
    `y`의 길이가 `tomotopy.models.SLDAModel.f`보다 짧을 경우, 모자란 값들은 자동으로 `NaN`으로 채워집니다.
''',
}

_UTILS_DOCS = {
    'Corpus': '''`Corpus`는 대량의 문헌을 간편하게 다룰 수 있게 도와주는 유틸리티 클래스입니다.
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
    `stopwords`가 호출가능한 경우, `stopwords(word) == True`이면 word는 불용어 처리되어 제외됩니다.''',
    'Corpus.add_doc': '''새 문헌을 코퍼스에 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.
이 메소드는 `words` 파라미터나 `raw`, `user_data` 파라미터 둘 중 하나를 요구합니다.
`words` 파라미터를 사용할 경우, `words`는 이미 전처리된 단어들의 리스트여야 합니다.
`raw` 파라미터를 사용할 경우, `raw`는 정제되기 전 문헌의 str이며, `tokenizer`가 이 비정제문헌을 처리하기 위해 호출됩니다.

만약 `tomotopy.models.DMRModel`의 `metadata`나 `tomotopy.models.SLDAModel`의 `y`처럼 특정한 토픽 모델에 필요한 추가 파라미터가 있다면 임의 키워드 인자로 넘겨줄 수 있습니다.

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
    추가적인 파라미터를 위한 임의 키워드 인자''',
    'Corpus.process': '''이터레이터 `data_feeder`를 통해 다수의 문헌을 코퍼스에 추가하고, 추가된 문헌의 개수를 반환합니다.

Parameters
----------
data_feeder : Iterable[Union[str, Tuple[str, Any], Tuple[str, Any, dict]]]
    문자열 `raw`이나, 튜플 (`raw`, `user_data`), 혹은 튜플 (`raw`, `user_data`, `kargs`) 를 반환하는 이터레이터. ''',
    'Corpus.save': '''현재 인스턴스를 파일 `filename`에 저장합니다.. 

Parameters
----------
filename : str
    인스턴스가 저장될 파일의 경로''',
    'Corpus.load': '''파일 `filename`로부터 인스턴스를 읽어들여 반환합니다.

Parameters
----------
filename : str
    읽어들일 파일의 경로''',
    'Corpus.extract_ngrams': '''..versionadded:: 0.10.0

PMI 점수를 이용해 자주 등장하는 n-gram들을 추출합니다.

Parameters
----------
min_cf : int
    추출할 n-gram의 최소 장서빈도
min_df : int
    추출할 n-gram의 최소 문헌빈도
max_len : int
    추출할 n-gram의 최대 길이
max_cand : int
    추출할 n-gram의 갯수
min_score : float
    추출할 n-gram의 최소 PMI 점수

Returns
-------
candidates : List[tomotopy.label.Candidate]
    추출된 n-gram 후보의 리스트. `tomotopy.label.Candidate` 타입''',
    'Corpus.concat_ngrams': '''..versionadded:: 0.10.0

코퍼스 내에서 주어진 n-gram 목록과 일치하는 단어열을 하나의 단어로 합칩니다.

Parameters
----------
cands : Iterable[tomotopy.label.Candidate]
    합칠 n-gram의 List. `tomotopy.utils.Corpus.extract_ngrams`로 생성할 수 있습니다.
delimiter : str
    여러 단어들을 연결할 때 사용할 구분자. 기본값은 `\'_\'`입니다.''',
    'SimpleTokenizer': '''`SimpleTokenizer`는 임의의 스테머를 사용할 수 있는 단순한 단어 분리 유틸리티입니다.

Parameters
----------
stemmer : Callable[str, str]
    단어를 스테밍하는데 사용되는 호출가능한 객체. 만약 이 값이 `None`이라면 스테밍은 사용되지 않습니다.
pattern : str
    토큰을 추출하는데 사용할 정규식 패턴
lowercase : bool
    참일 경우 분리된 단어들을 소문자화합니다.

SimpleTokenizer와 NLTK를 사용하여 스테밍을 하는 예제는 다음과 같습니다.

.. include:: ./auto_labeling_code_with_porter.rst''',
}

_COHERENCE_DOCS = {
    'Coherence': '''`Coherence` 클래스는 coherence를 계산하는 방법을 제공합니다.

주어진 코퍼스를 바탕으로 coherence를 계산하는 인스턴스를 초기화합니다.

Parameters
----------
corpus : Union[tomotopy.utils.Corpus, tomotopy.models.LDAModel]
    단어 분포 확률을 추정하기 위한 레퍼런스 코퍼스.
    `tomotopy.utils.Corpus` 타입뿐만 아니라 `tomotopy.models.LDAModel`를 비롯한 다양한 토픽 모델링 타입의 인스턴스까지 지원합니다.
    만약 `corpus`가 `tomotpy.utils.Corpus`의 인스턴스라면 `targets`이 반드시 주어져야 합니다.
coherence : Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
    coherence를 계산하는 데 사용될 척도. 척도는 (`tomotopy.coherence.ProbEstimation`, `tomotopy.coherence.Segmentation`, `tomotopy.coherence.ConfirmMeasure`)의 조합이거나
    (`tomotopy.coherence.ProbEstimation`, `tomotopy.coherence.Segmentation`, `tomotopy.coherence.ConfirmMeasure`, `tomotopy.coherence.IndirectMeasure`)의 조합이어야 합니다.
    
    또한 다음과 같이 `str` 타입의 단축표현도 제공됩니다.
    > * \'u_mass\' : (`tomotopy.coherence.ProbEstimation.DOCUMENT`, `tomotopy.coherence.Segmentation.ONE_PRE`, `tomotopy.coherence.ConfirmMeasure.LOGCOND`)
    > * \'c_uci\' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_ONE`, `tomotopy.coherence.ConfirmMeasure.PMI`)
    > * \'c_npmi\' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_ONE`, `tomotopy.coherence.ConfirmMeasure.NPMI`)
    > * \'c_v\' : (`tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`, `tomotopy.coherence.Segmentation.ONE_SET`, `tomotopy.coherence.ConfirmMeasure.NPMI`, `tomotopy.coherence.IndirectMeasure.COSINE`)
window_size : int
    `tomotopy.coherence.ProbEstimation.SLIDING_WINDOWS`가 사용될 경우 쓰일 window 크기.
    기본값은 \'c_uci\'와 \'c_npmi\'의 경우 10, \'c_v\'의 경우 110입니다.
targets : Iterable[str]
    만약 `corpus`가 `tomotpy.utils.Corpus`의 인스턴스인 경우, 목표 단어가 주어져야 합니다. 
    `targets`에 주어진 단어 목록에 대해서만 확률 분포가 추정됩니다.
top_n : int
    각 토픽에서 추출할 상위 단어의 개수.
    만약 `corpus`이 `tomotopy.models.LDAModel`나 기타 토픽 모델의 인스턴스인 경우, 목표 단어는 각 토픽의 상위 단어에서 추출됩니다.
    만약 `targets`이 주어진 경우 `corpus`가 토픽 모델인 경우에도 `targets`에서 목표 단어를 가져옵니다.
eps : float
    계산 과정에서 0으로 나누는 것을 방지하기 위한 epsilon 값
gamma : float
    indirect confirm measure 계산에 사용되는 gamma 값''',
    'Coherence.get_score': '''주어진 `words` 또는 `topic_id`를 이용해 coherence를 계산합니다.

Parameters
----------
words : Iterable[str]
    coherence가 계산될 단어들.
    만약 `tomotopy.coherence.Coherence`가 `tomotopy.models.LDAModel`나 기타 토픽 모델의 인스턴스로 `corpus`를 받아 초기화된 경우 `words`는 생략될 수 있습니다.
    이 경우 단어들은 토픽 모델의 `topic_id` 토픽에서 추출됩니다.
topic_id : int
    단어가 추출될 토픽의 id.
    이 파라미터는 오직 `tomotopy.coherence.Coherence`가 `tomotopy.models.LDAModel`나 기타 토픽 모델의 인스턴스로 `corpus`를 받아 초기화된 경우에만 사용 가능합니다.
    생략시 모든 토픽의 coherence 점수를 평균낸 값이 반환됩니다.
timepoint : int
    ..versionadded:: 0.12.3

    단어가 추출될 토픽의 시점 (`tomotopy.models.DTModel`에서만 유효)''',
}


def apply():
    """Apply Korean docstring overrides."""
    import tomotopy
    from tomotopy import utils, coherence, label
    from tomotopy.utils import Document, Corpus, SimpleTokenizer
    from tomotopy.label import Candidate, PMIExtractor, FoRelevance
    from tomotopy.coherence import (
        ProbEstimation, Segmentation, ConfirmMeasure, IndirectMeasure, Coherence
    )
    from tomotopy.models import (
        LDAModel, DMRModel, GDMRModel, HDPModel, MGLDAModel,
        PAModel, HPAModel, CTModel, SLDAModel, LLDAModel,
        PLDAModel, HLDAModel, DTModel, PTModel,
    )

    # Module docs
    _mod_map = {
        'tomotopy': tomotopy,
        'tomotopy.utils': utils,
        'tomotopy.coherence': coherence,
        'tomotopy.label': label,
    }
    for path, doc in _MODULE_DOCS.items():
        _mod_map[path].__doc__ = doc

    # Enum/top-level docs
    _enum_map = {
        'TermWeight': tomotopy.TermWeight,
        'ParallelScheme': tomotopy.ParallelScheme,
        'ProbEstimation': ProbEstimation,
        'Segmentation': Segmentation,
        'ConfirmMeasure': ConfirmMeasure,
        'IndirectMeasure': IndirectMeasure,
    }
    for key, doc in _DOCS.items():
        if '.' not in key:
            if key in _enum_map:
                _enum_map[key].__doc__ = doc
            continue
        cls_name, attr = key.split('.', 1)
        if cls_name in _enum_map:
            try:
                member = _enum_map[cls_name][attr]
                member.__doc__ = doc
            except (KeyError, AttributeError):
                pass

    # Class/method/property docs from C++ (docs.h/label_docs.h)
    _cls_map = {
        'Document': Document, 'LDAModel': LDAModel, 'DMRModel': DMRModel,
        'GDMRModel': GDMRModel, 'HDPModel': HDPModel, 'MGLDAModel': MGLDAModel,
        'PAModel': PAModel, 'HPAModel': HPAModel, 'CTModel': CTModel,
        'SLDAModel': SLDAModel, 'LLDAModel': LLDAModel, 'PLDAModel': PLDAModel,
        'HLDAModel': HLDAModel, 'DTModel': DTModel, 'PTModel': PTModel,
        'Candidate': Candidate, 'PMIExtractor': PMIExtractor,
        'FoRelevance': FoRelevance,
        'Corpus': Corpus, 'SimpleTokenizer': SimpleTokenizer,
        'Coherence': Coherence,
    }
    _apply_docs(_DOCS, _cls_map)
    _apply_docs(_UTILS_DOCS, _cls_map)
    _apply_docs(_COHERENCE_DOCS, _cls_map)


def _apply_docs(docs_dict, cls_map):
    for key, doc in docs_dict.items():
        if '.' not in key:
            cls = cls_map.get(key)
            if cls is not None:
                cls.__doc__ = doc
                # Also apply to __init__ if it exists in the class
                init = vars(cls).get('__init__')
                if init is not None and hasattr(init, '__doc__'):
                    try:
                        init.__doc__ = doc
                    except (AttributeError, TypeError):
                        pass
            continue
        cls_name, attr = key.split('.', 1)
        cls = cls_map.get(cls_name)
        if cls is None:
            continue
        # Try as a descriptor in class __dict__ (property)
        obj = vars(cls).get(attr)
        if obj is not None and hasattr(obj, "__doc__"):
            try:
                obj.__doc__ = doc
                continue
            except (AttributeError, TypeError):
                pass
        # Try as a regular attribute
        obj = getattr(cls, attr, None)
        if obj is not None and hasattr(obj, "__doc__"):
            try:
                obj.__doc__ = doc
            except (AttributeError, TypeError):
                pass
