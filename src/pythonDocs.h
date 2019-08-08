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
	"get_topics(self, top_n=10)",
	u8R""(Return the `top_n` topics with its probability of the document.)"",
	u8R""(현재 문헌의 상위 `top_n`개의 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_topic_dist__doc__,
	"get_topic_dist(self)",
	u8R""(Return a distribution of the topics in the document.)"",
	u8R""(현재 문헌의 토픽 확률 분포를 `list` 형태로 반환합니다.)"");

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

DOC_VARIABLE_EN_KO(Document_metadata__doc__,
	u8R""("metadata of document (for only `tomotopy.DMRModel` model, read-only))"",
	u8R""(문헌의 메타데이터 (`tomotopy.DMRModel` 모형에서만 사용됨, 읽기전용))"");

DOC_VARIABLE_EN_KO(Document_subtopics__doc__,
	u8R""("a `list` of sub topics for each word (for only `tomotopy.PAModel` and `tomotopy.HPAModel` model, read-only)")"",
	u8R""(문헌의 단어들이 각각 할당된 하위 토픽을 보여주는 `list` (`tomotopy.PAModel`와 `tomotopy.HPAModel` 모형에서만 사용됨, 읽기전용))"");

DOC_VARIABLE_EN_KO(Document_windows__doc__,
	u8R""(a `list` of window IDs for each word (for only `tomotopy.MGLDAModel` model, read-only))"",
	u8R""(문헌의 단어들이 할당된 윈도우의 ID를 보여주는 `list` (`tomotopy.MGLDAModel` 모형에서만 사용됨, 읽기전용))"");

/*
	class LDA
*/
DOC_SIGNATURE_EN_KO(LDA___init____doc__,
	"LDAModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, seed=None)",
	u8R""(This type provides Latent Dirichlet Allocation(LDA) topic model and its implementation is based on following papers:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.
	
LDAModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k` : the number of topics between 1 ~ 32767.
* `alpha` : hyperparameter of Dirichlet distribution for document-topic
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. The default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Latent Dirichlet Allocation(LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.
	
LDAModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, seed=?)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `k` : 토픽의 개수, 1 ~ 32767 범위의 정수.
* `alpha` : 문헌-토픽 디리클레 분포의 하이퍼 파라미터
* `eta` : 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_add_doc__doc__,
	"add_doc(self, words)",
	u8R""(Add a new document into the model instance and return an index of the inserted document.

* `words` : an iterable of `str`
)"",
u8R""(현재 모델에 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable)"");

DOC_SIGNATURE_EN_KO(LDA_make_doc__doc__,
	"make_doc(self, words)",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iterable of `str`
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다..

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable
)"");

DOC_SIGNATURE_EN_KO(LDA_train__doc__,
	"train(self, iter=10, workers=0)",
	u8R""(Train the model using Gibbs-sampling with `iter` iterations. Return `None`. 
After calling this method, you cannot `tomotopy.LDAModel.add_doc` more.

* `iter` : the number of iterations of Gibbs-sampling
* `workers` : an integer indicating the number of workers to perform samplings. 
If `workers` is 0, the number of cores in the system will be used.
)"",
u8R""(깁스 샘플링을 `iter` 회 반복하여 현재 모델을 학습시킵니다. 반환값은 `None`입니다. 
이 메소드가 호출된 이후에는 더 이상 `tomotopy.LDAModel.add_doc`로 현재 모델에 새로운 학습 문헌을 추가시킬 수 없습니다.

* `iter` : 깁스 샘플링의 반복 횟수
* `workers` : 깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer, indicating the topic, in range [0, `k`)
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

* `topic_id` : 토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(LDA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

* `topic_id` : an integer, indicating the topic, in range [0, `k`)
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

* `topic_id` : 토픽을 가리키는 [0, `k`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(LDA_get_count_by_topics__doc__,
	"get_count_by_topics(self)",
	u8R""(Return the number of words allocated to each topic.)"",
	u8R""(각각의 토픽에 할당된 단어의 개수를 `list`형태로 반환합니다.)"");

DOC_SIGNATURE_EN_KO(LDA_infer__doc__,
	"infer(self, document, iter=100, tolerance=-1, workers=0, together=False)",
	u8R""(Return the inferred topic distribution from unseen `document`s.
The return type is (a topic distribution of `document`, log likelihood) or (a `list` of topic distribution of `document`, log likelihood)

* `document` : an instance of `tomotopy.Document` or a `list` of instances of `tomotopy.Document` to be inferred by the model.
It can be acquired from `tomotopy.LDAModel.make_doc` method.
* `iter` : an integer indicating the number of iteration to estimate the distribution of topics of `document`.
The higher value will generate a more accuracy result.
* `tolerance` isn't currently used.
* `workers` : an integer indicating the number of workers to perform samplings. 
If `workers` is 0, the number of cores in the system will be used.
* `together` : all `document`s are infered together in one process if True, otherwise each `document` is infered independently. Its default value is `False`.
)"",
u8R""(새로운 문헌인 `document`에 대해 각각의 주제 분포를 추론하여 반환합니다.
반환 타입은 (`document`의 주제 분포, 로그가능도) 또는 (`document`의 주제 분포로 구성된 `list`, 로그가능도)입니다.

* `document` : 추론에 사용할 `tomotopy.Document`의 인스턴스이거나 이 인스턴스들의 `list`.
이 인스턴스들은 `tomotopy.LDAModel.make_doc` 메소드를 통해 얻을 수 있습니다.
* `iter` : `document`의 주제 분포를 추론하기 위해 학습을 반복할 횟수입니다.
이 값이 클 수록 더 정확한 결과를 낼 수 있습니다.
* `tolerance` : 현재는 사용되지 않음
* `workers` : 깁스 샘플링을 수행하는 데에 사용할 스레드의 개수입니다. 
만약 이 값을 0으로 설정할 경우 시스템 내의 가용한 모든 코어가 사용됩니다.
* `together` : 이 값이 True인 경우 입력한 `document` 문헌들을 한 번에 모델에 넣고 추론을 진행합니다.
False인 경우 각각의 문헌들을 별도로 모델에 넣어 추론합니다. 기본값은 `False`입니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_save__doc__,
	"save(self, filename, full=True)",
	u8R""(Save the model instance to file `filename`. Return `None`.

If `full` is `True`, the model with its all documents and state will be saved. If you want to train more after, use full model.
If `False`, only topic paramters of the model will be saved. This model can be only used for inference of an unseen document.
)"",
u8R""(현재 모델을 `filename` 경로의 파일에 저장합니다. `None`을 반환합니다.

`full`이 `True`일 경우, 모델의 전체 상태가 파일에 모두 저장됩니다. 저장된 모델을 다시 읽어들여 학습(`train`)을 더 진행하고자 한다면 `full` = `True`로 하여 저장하십시오.
반면 `False`일 경우, 토픽 추론에 관련된 파라미터만 파일에 저장됩니다. 이 경우 파일의 용량은 작아지지만, 추가 학습은 불가하고 새로운 문헌에 대해 추론(`infer`)하는 것만 가능합니다.
)"");

DOC_SIGNATURE_EN_KO(LDA_load__doc__,
	"load(filename)",
	u8R""(Return the model instance loaded from file `filename`.)"",
	u8R""(`filename` 경로의 파일로부터 모델 인스턴스를 읽어들여 반환합니다.)"");


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
	u8R""(the hyperparameter alpha (read-only))"",
	u8R""(하이퍼 파라미터 alpha (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_eta__doc__,
	u8R""(the hyperparameter eta (read-only))"",
	u8R""(하이퍼 파라미터 eta (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_docs__doc__,
	u8R""(a `list`-like interface of `tomotopy.Document` in the model instance (read-only))"",
	u8R""(현재 모델에 포함된 `tomotopy.Document`에 접근할 수 있는 `list`형 인터페이스 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_vocabs__doc__,
	u8R""(a dictionary of vocabuluary as type `tomotopy.Dictionary` (read-only))"",
	u8R""(현재 모델에 포함된 어휘들을 보여주는 `tomotopy.Dictionary` 타입의 어휘 사전 (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_num_vocabs__doc__,
	u8R""(the number of vocabuluaries after words with a smaller frequency were removed (read-only)

This value is 0 before `train` called.)"",
	u8R""(작은 빈도의 단어들을 제거한 뒤 남은 어휘의 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.)"");

DOC_VARIABLE_EN_KO(LDA_vocab_freq__doc__,
	u8R""(a `list` of vocabulary frequencies included in the model (read-only))"",
	u8R""(현재 모델에 포함된 어휘들의 빈도를 보여주는 `list` (읽기전용))"");

DOC_VARIABLE_EN_KO(LDA_num_words__doc__,
	u8R""(the number of total words (read-only)

This value is 0 before `train` called.)"",
	u8R""(현재 모델에 포함된 문헌들 전체의 단어 개수 (읽기전용)

`train`이 호출되기 전에는 이 값은 0입니다.)"");

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


/*
	class DMR
*/
DOC_SIGNATURE_EN_KO(DMR___init____doc__,
	"DMRModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=0.0000000001, seed=None)",
	u8R""(This type provides Dirichlet Multinomial Regression(DMR) topic model and its implementation is based on following papers:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

DMRModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=1e-10, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k` : the number of topics between 1 ~ 32767
* `alpha` : exponential of mean of normal distribution for `lambdas`
* `eta` : hyperparameter of Dirichlet distribution for topic - word
* `sigma` : standard deviation of normal distribution for `lambdas`
* `alpha_epsilon` : small smoothing value for preventing `exp(lambdas)` to be zero
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Dirichlet Multinomial Regression(DMR) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

DMRModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=1e-10, seed=None)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `k` : 토픽의 개수, 1 ~ 32767 범위의 정수.
* `alpha` : 문헌-토픽 디리클레 분포의 하이퍼 파라미터
* `eta` : 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `sigma` : `lambdas` 파라미터의 표준 편차
* `alpha_epsilon` : `exp(lambdas)`가 0이 되는 것을 방지하는 평탄화 계수
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(DMR_add_doc__doc__,
	"add_doc(self, words, metadata='')",
	u8R""(Add a new document into the model instance with `metadata` and return an index of the inserted document.

* `words` : an iterable of `str`
* `metadata` : a `str` indicating metadata of the document (e.g., author, title or year)
)"",
u8R""(현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable
* `metadata` : 문헌의 메타데이터를 표현하는 `str` (예를 들어 저자나 제목, 작성연도 등)
)"");

DOC_SIGNATURE_EN_KO(DMR_make_doc__doc__,
	"make_doc(self, words, metadata='')",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iteratable of `str`
* `metadata` : a `str` indicating metadata of the document (e.g., author, title or year)
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable
* `metadata` : 문헌의 메타데이터를 표현하는 `str` (예를 들어 저자나 제목, 작성연도 등)
)"");

DOC_VARIABLE_EN_KO(DMR_f__doc__,
	u8R""(the number of metadata features (read-only))"",
	u8R""(메타데이터 자질 종류의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_sigma__doc__,
	u8R""(the hyperparamter sigma (read-only))"",
	u8R""(하이퍼 파라미터 sigma (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_alpha_epsilon__doc__,
	u8R""(the smooting value alpha-epsilon (read-only))"",
	u8R""(평탄화 계수 alpha-epsilon (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_metadata_dict__doc__,
	u8R""(a dictionary of metadata in type `tomotopy.Dictionary` (read-only))"",
	u8R""(`tomotopy.Dictionary` 타입의 메타데이터 사전 (읽기전용))"");

DOC_VARIABLE_EN_KO(DMR_lamdas__doc__,
	u8R""(a `list` of paramter lambdas (read-only))"",
	u8R""(현재 모형의 lambda 파라미터를 보여주는 `list` (읽기전용))"");

/*
	class HDP
*/
DOC_SIGNATURE_EN_KO(HDP___init____doc__,
	"HDPModel(tw=TermWeight.ONE, min_cf=0, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=None)",
	u8R""(This type provides Hierarchical Dirichlet Process(HDP) topic model and its implementation is based on following papers:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

HDPModel(tw=TermWeight.ONE, min_cf=0, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `initial_k` : the initial number of topics between 1 ~ 32767.
The number of topics will be adjusted for data during training.
* `alpha` : concentration coeficient of Dirichlet Process for document-table 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `gamma` : concentration coeficient of Dirichlet Process for table-topic
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Hierarchical Dirichlet Process(HDP) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

HDPModel(tw=TermWeight.ONE, min_cf=0, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=None)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `initial_k` : 초기 토픽의 개수를 지정하는 1 ~ 32767 범위의 정수.
토픽 개수는 학습 과정에서 최적의 수치로 조정됩니다.
* `alpha` : document-table에 대한 Dirichlet Process의 집중 계수
* `eta` : 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `gamma` : table-topic에 대한 Dirichlet Process의 집중 계수
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(HDP_is_live_topic__doc__,
	"is_live_topic(self, topic_id)",
	u8R""(Return `True` if the topic `topic_id` is alive, otherwise return `False`.

* `topic_id` : an integer in range [0, `k`) indicating the topic
)"",
u8R""(`topic_id`가 유효한 토픽을 가리키는 경우 `True`, 아닌 경우 `False`를 반환합니다.

* `topic_id` : 토픽을 가리키는 [0, `k`) 범위의 정수
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
	"MGLDAModel(tw=TermWeight.ONE, min_cf=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=None)",
	u8R""(This type provides Multi Grain Latent Dirichlet Allocation(MG-LDA) topic model and its implementation is based on following papers:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

MGLDAModel(tw=TermWeight.ONE, min_cf=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k_g` : the number of global topics between 1 ~ 32767
* `k_l` : the number of local topics between 1 ~ 32767
* `t` : the size of sentence window
* `alpha_g` : hyperparameter of Dirichlet distribution for document-global topic
* `alpha_l` : hyperparameter of Dirichlet distribution for document-local topic
* `alpha_mg` : hyperparameter of Dirichlet distribution for global-local selection (global coef)
* `alpha_ml` : hyperparameter of Dirichlet distribution for global-local selection (local coef)
* `eta_g` : hyperparameter of Dirichlet distribution for global topic-word
* `eta_l` : hyperparameter of Dirichlet distribution for local topic-word
* `gamma` : hyperparameter of Dirichlet distribution for sentence-window
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Multi Grain Latent Dirichlet Allocation(MG-LDA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

MGLDAModel(tw=TermWeight.ONE, min_cf=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=None)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `k_g` : 전역 토픽의 개수를 지정하는 1 ~ 32767 사이의 정수
* `k_l` : 지역 토픽의 개수를 지정하는 1 ~ 32767 사이의 정수
* `t` : 문장 윈도우의 크기
* `alpha_g` : 문헌-전역 토픽 디리클레 분포의 하이퍼 파라미터
* `alpha_l` : 문헌-지역 토픽 디리클레 분포의 하이퍼 파라미터
* `alpha_mg` : 전역/지역 선택 디리클레 분포의 하이퍼 파라미터 (전역 부분 계수)
* `alpha_ml` : 전역/지역 선택 디리클레 분포의 하이퍼 파라미터 (지역 부분 계수)
* `eta_g` : 전역 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `eta_l` : 지역 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `gamma` : 문장-윈도우 디리클레 분포의 하이퍼 파라미터
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_add_doc__doc__,
	"add_doc(self, words, delimiter='.')",
	u8R""(Add a new document into the model instance and return an index of the inserted document.

* `words` : an iterable of `str`
* `delimiter` : a sentence separator. `words` will be separated by this value into sentences.
)"",
u8R""(현재 모델에 `metadata`를 포함하는 새로운 문헌을 추가하고 추가된 문헌의 인덱스 번호를 반환합니다.

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable
* `delimiter` : 문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_make_doc__doc__,
	"make_doc(self, words, delimiter='.')",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iteratable of `str`
* `delimiter` : a sentence separator. `words` will be separated by this value into sentences.
)"",
u8R""(`words` 단어를 바탕으로 새로운 문헌인 `tomotopy.Document` 인스턴스를 반환합니다. 이 인스턴스는 `tomotopy.LDAModel.infer` 메소드에 사용될 수 있습니다.

* `words` : 문헌의 각 단어를 나열하는 `str` 타입의 iterable
* `delimiter` : 문장 구분자, `words`는 이 값을 기준으로 문장 단위로 반할됩니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer. 
A number in range [0, `k_g`) indicates a global topic and 
a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

* `topic_id` : [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
)"");

DOC_SIGNATURE_EN_KO(MGLDA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current topic.

* `topic_id` : an integer. 
A number in range [0, `k_g`) indicates a global topic and 
a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

* `topic_id` : [0, `k_g`) 범위의 정수는 전역 토픽을, [`k_g`, `k_g` + `k_l`) 범위의 정수는 지역 토픽을 가리킵니다.
)"");

DOC_VARIABLE_EN_KO(MGLDA_k_g__doc__,
	u8R""(the hyperparamter k_g (read-only))"",
	u8R""(하이퍼 파라미터 k_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_k_l__doc__,
	u8R""(the hyperparamter k_l (read-only))"",
	u8R""(하이퍼 파라미터 k_l (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_gamma__doc__,
	u8R""(the hyperparamter gamma (read-only))"",
	u8R""(하이퍼 파라미터 gamma (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_t__doc__,
	u8R""(the hyperparamter t (read-only))"",
	u8R""(하이퍼 파라미터 t (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_g__doc__,
	u8R""(the hyperparamter alpha_g (read-only))"",
	u8R""(하이퍼 파라미터 alpha_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_l__doc__,
	u8R""(the hyperparamter alpha_l (read-only))"",
	u8R""(하이퍼 파라미터 alpha_l (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_mg__doc__,
	u8R""(the hyperparamter alpha_mg (read-only))"",
	u8R""(하이퍼 파라미터 alpha_mg (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_alpha_ml__doc__,
	u8R""(the hyperparamter alpha_ml (read-only))"",
	u8R""(하이퍼 파라미터 alpha_ml (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_eta_g__doc__,
	u8R""(the hyperparamter eta_g (read-only))"",
	u8R""(하이퍼 파라미터 eta_g (읽기전용))"");

DOC_VARIABLE_EN_KO(MGLDA_eta_l__doc__,
	u8R""(the hyperparamter eta_l (read-only))"",
	u8R""(하이퍼 파라미터 eta_l (읽기전용))"");


/*
	class PA
*/
DOC_SIGNATURE_EN_KO(PA___init____doc__,
	"PAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)",
	u8R""(This type provides Pachinko Allocation(PA) topic model and its implementation is based on following papers:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

PAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k1` : the number of super topics between 1 ~ 32767
* `k2` : the number of sub topics between 1 ~ 32767
* `alpha` : initial hyperparameter of Dirichlet distribution for document-super topic 
* `eta` : hyperparameter of Dirichlet distribution for sub topic-word
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Pachinko Allocation(PA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

PAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `k1` : 상위 토픽의 개수, 1 ~ 32767 사이의 정수.
* `k2` : 하위 토픽의 개수, 1 ~ 32767 사이의 정수.
* `alpha` : 문헌-상위 토픽 디리클레 분포의 하이퍼 파라미터
* `eta` : 하위 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(PA_get_topic_words__doc__,
	"get_topic_words(self, sub_topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the sub topic `sub_topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `sub_topic_id` : an integer, indicating the sub topic, in range [0, `k2`)
)"",
u8R""(하위 토픽 `sub_topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

* `sub_topic_id` : 하위 토픽을 가리키는 [0, `k2`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(PA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, sub_topic_id)",
	u8R""(Return the word distribution of the sub topic `sub_topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in the current sub topic.

* `sub_topic_id` : an integer, indicating the sub topic, in range [0, `k2`)
)"",
u8R""(하위 토픽 `sub_topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 하위 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

* `sub_topic_id` : 하위 토픽을 가리키는 [0, `k2`) 범위의 정수
)"");

DOC_SIGNATURE_EN_KO(PA_get_sub_topics__doc__,
	"get_sub_topics(self, super_topic_id, top_n=10)",
	u8R""(Return the `top_n` sub topics and its probability in a super topic `super_topic_id`.
The return type is a `list` of (subtopic:`int`, probability:`float`).

* `super_topic_id` : an integer, indicating the super topic, in range [0, `k1`)

.. versionadded:: 0.1.4
)"",
u8R""(상위 토픽 `super_topic_id`에 속하는 상위 `top_n`개의 하위 토픽과 각각의 확률을 반환합니다. 
반환 타입은 (하위토픽:`int`, 확률:`float`) 튜플의 `list`형입니다.

* `super_topic_id` : 상위 토픽을 가리키는 [0, `k1`) 범위의 정수

.. versionadded:: 0.1.4
)"");

DOC_SIGNATURE_EN_KO(PA_get_sub_topic_dist__doc__,
	"get_sub_topic_dist(self, super_topic_id)",
	u8R""(Return a distribution of the sub topics in a super topic `super_topic_id`.
The returned value is a `list` that has `k2` fraction numbers indicating probabilities for each sub topic in the current super topic.

* `super_topic_id` : an integer, indicating the super topic, in range [0, `k1`)
)"",
u8R""(상위 토픽 `super_topic_id`의 하위 토픽 분포를 반환합니다.
반환하는 값은 현재 상위 토픽 내 각각의 하위 토픽들의 발생확률을 나타내는 `k2`개의 소수로 구성된 `list`입니다.

* `super_topic_id` : 상위 토픽을 가리키는 [0, `k1`) 범위의 정수
)"");

DOC_VARIABLE_EN_KO(PA_k1__doc__,
	u8R""(k1, the number of super topics (read-only))"",
	u8R""(k1, 상위 토픽의 개수 (읽기전용))"");

DOC_VARIABLE_EN_KO(PA_k2__doc__,
	u8R""(k2, the number of sub topics (read-only))"",
	u8R""(k2, 하위 토픽의 개수 (읽기전용))"");

/*
	class HPA
*/
DOC_SIGNATURE_EN_KO(HPA___init____doc__,
	"HPAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)",
	u8R""(This type provides Hierarchical Pachinko Allocation(HPA) topic model and its implementation is based on following papers:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

HPAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k1` : the number of super topics between 1 ~ 32767
* `k2` : the number of sub topics between 1 ~ 32767
* `alpha` : initial hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"",
u8R""(이 타입은 Hierarchical Pachinko Allocation(HPA) 토픽 모델의 구현체를 제공합니다. 주요 알고리즘은 다음 논문에 기초하고 있습니다:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

HPAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=None)

* `tw` : 용어 가중치 기법을 나타내는 `tomotopy.TermWeight`의 열거값. 기본값은 TermWeight.ONE 입니다.
* `min_cf` : 단어의 최소 장서 빈도. 전체 문헌 내의 출현 빈도가 `min_cf`보다 작은 단어들은 모델에서 제외시킵니다.
기본값은 0으로, 이 경우 어떤 단어도 제외되지 않습니다.
* `k1` : 상위 토픽의 개수, 1 ~ 32767 사이의 정수.
* `k2` : 하위 토픽의 개수, 1 ~ 32767 사이의 정수.
* `alpha` : 문헌-상위 토픽 디리클레 분포의 하이퍼 파라미터
* `eta` : 하위 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN_KO(HPA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer. 
0 indicates the top topic, 
a number in range [1, 1 + `k1`) indicates a super topic and
a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
)"",
u8R""(토픽 `topic_id`에 속하는 상위 `top_n`개의 단어와 각각의 확률을 반환합니다. 
반환 타입은 (단어:`str`, 확률:`float`) 튜플의 `list`형입니다.

* `topic_id` : 0일 경우 최상위 토픽을 가리키며,
[1, 1 + `k1`) 범위의 정수는 상위 토픽을,
[1 + `k1`, 1 + `k1` + `k2`) 범위의 정수는 하위 토픽을 가리킵니다.
)"");

DOC_SIGNATURE_EN_KO(HPA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in current topic.

* `topic_id` : an integer. 
0 indicates the top topic, 
a number in range [1, 1 + `k1`) indicates a super topic and
a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
)"",
u8R""(토픽 `topic_id`의 단어 분포를 반환합니다.
반환하는 값은 현재 하위 토픽 내 각각의 단어들의 발생확률을 나타내는 `len(vocabs)`개의 소수로 구성된 `list`입니다.

* `topic_id` : 0일 경우 최상위 토픽을 가리키며,
[1, 1 + `k1`) 범위의 정수는 상위 토픽을,
[1 + `k1`, 1 + `k1` + `k2`) 범위의 정수는 하위 토픽을 가리킵니다.
)"");
