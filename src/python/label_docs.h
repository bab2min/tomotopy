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
	class Candidate
*/
DOC_SIGNATURE_EN_KO(Candidate___init____doc__,
	"Candidate()",
	u8R""()"",
u8R""()"");

DOC_VARIABLE_EN_KO(Candidate_words__doc__,
	u8R""(words of the candidate for topic label (read-only))"",
	u8R""(토픽 레이블의 후보 (읽기전용))"");

DOC_VARIABLE_EN_KO(Candidate_score__doc__,
	u8R""(score of the candidate (read-only))"",
	u8R""(후보의 점수(읽기전용))"");

DOC_VARIABLE_EN_KO(Candidate_name__doc__,
	u8R""(an actual name of the candidate for topic label)"",
	u8R""(토픽 레이블의 실제 이름)"");

/*
	class PMIExtractor
*/
DOC_SIGNATURE_EN_KO(Extractor_extract__doc__,
	"extract(self, topic_model)",
	u8R""(Return the list of `tomotopy.label.Candidate`s extracted from `topic_model`

Parameters
----------
topic_model
    an instance of topic model with documents to extract candidates
)"",
	u8R""(`topic_model`로부터 추출된 토픽 레이블 후보인 `tomotopy.label.Candidate`의 리스트를 반환합니다.

Parameters
----------
topic_model
    후보를 추출할 문헌들을 가지고 있는 토픽 모델의 인스턴스
)"");

DOC_SIGNATURE_EN_KO(PMIExtractor___init____doc__,
	"PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=5000)",
	u8R""(.. versionadded:: 0.6.0

`PMIExtractor` exploits multivariate pointwise mutual information to extract collocations. 
It finds a string of words that often co-occur statistically.

Parameter
---------
min_cf : int
    minimum collection frequency of collocations. Collocations with a smaller collection frequency than `min_cf` are excluded from the candidates.
    Set this value large if the corpus is big
min_df : int
    minimum document frequency of collocations. Collocations with a smaller document frequency than `min_df` are excluded from the candidates.
    Set this value large if the corpus is big
max_len : int
    maximum length of collocations
max_cand : int
    maximum number of candidates to extract
)"",
	u8R""(.. versionadded:: 0.6.0

`PMIExtractor`는 다변수 점별 상호정보량을 활용해 연어를 추출합니다. 이는 통계적으로 자주 함께 등장하는 단어열을 찾아줍니다.

Parameter
---------
min_cf : int
    추출하려는 후보의 최소 장서 빈도. 문헌 내 등장하는 빈도수가 `min_cf`보다 작은 연어는 후보에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
min_df : int
    추출하려는 후보의 최소 문헌 빈도. 연어가 등장하는 문헌 수가 `min_df`보다 작은 경우 후보에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
max_len : int
    분석하려는 연어의 최대 길이
max_cand : int
    추출하려는 후보의 최대 개수
)"");

/*
	class FoRelevance
*/
DOC_SIGNATURE_EN_KO(Labeler_get_topic_labels__doc__,
	"get_topic_labels(self, k, top_n=10)",
	u8R""(Return the top-n label candidates for the topic `k`

Parameter
---------
k : int
    an integer indicating a topic
top_n : int
    the number of labels
)"",
	u8R""(토픽 `k`에 해당하는 레이블 후보 상위 n개를 반환합니다.

Parameter
---------
k : int
    토픽을 지정하는 정수
top_n : int
    토픽 레이블의 개수)"");

DOC_SIGNATURE_EN_KO(FoRelevance___init____doc__,
	"FoRelevance(topic_model, cands, min_df=5, smoothing=0.01, mu=0.25, workers=0)",
	u8R""(.. versionadded:: 0.6.0

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
workers : int
    an integer indicating the number of workers to perform samplings. 
    If `workers` is 0, the number of cores in the system will be used.
)"",
	u8R""(.. versionadded:: 0.6.0

First-order Relevance를 이용한 토픽 라벨링 기법을 제공합니다. 이 구현체는 다음 논문에 기초하고 있습니다:

> * Mei, Q., Shen, X., & Zhai, C. (2007, August). Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499).

Parameters
----------
topic_model
    토픽명을 붙일 토픽 모델의 인스턴스
cands : Iterable[tomotopy.label.Candidate]
    토픽명으로 사용될 후보들의 리스트
min_df : int
    사용하려는 후보의 최소 문헌 빈도. 연어가 등장하는 문헌 수가 `min_cf`보다 작은 경우 선택에서 제외됩니다.
    분석하려는 코퍼스가 클 경우 이 값을 키우십시오.
smoothing : float
    라플라스 평활화에 사용될 0보다 큰 작은 실수
mu : float
    변별성 계수. 특정 토픽에 대해서만 높은 점수를 가지고, 나머지 토픽에 대해서는 낮은 점수를 가진 후보는 이 계수가 클 때 더 높은 최종 점수를 받습니다.
workers : int
    깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
    만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
)"");
