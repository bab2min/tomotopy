#pragma once
#include <Python.h>

#define DOC_SIGNATURE_EN(name, signature, en) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#ifdef DOC_KO
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" ko)
#else
#define DOC_SIGNATURE_EN_KO(name, signature, en, ko) PyDoc_STRVAR(name, signature "\n--\n\n" en)
#endif

/*
	class Document
*/
DOC_SIGNATURE_EN_KO(Document___init____doc__,
	"Document()",
	u8R""(This type provides abstract model to access documents to be used Topic Model.

An instance of this type can be acquired from `tomotopy.LDAModel.make_doc` method or `tomotopy.LDAModel.docs` member of each Topic Model instance.)"",
	u8R""(이 타입은 토픽 모델에 사용될 문헌들에 접근할 수 있도록 하는 추상 모형을 제공합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_topics__doc__,
	"get_topics(self, top_n=10)",
	u8R""(Return the `top_n` topics with its probability of the document.)"",
	u8R""(현재 문헌의 상위 `top_n`개의 토픽과 그 확률을 `tuple`의 `list` 형태로 반환합니다.)"");

DOC_SIGNATURE_EN_KO(Document_get_topic_dist__doc__,
	"get_topic_dist(self)",
	u8R""(Return a distribution of the topics in the document.)"",
	u8R""(현재 문헌의 토픽 확률 분포를 `list` 형태로 반환합니다.)"");

/*
	class LDA
*/
DOC_SIGNATURE_EN_KO(LDA___init____doc__,
	"LDAModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, seed=?)",
	u8R""(This type provides Latent Dirichlet Allocation(LDA) topic model and its implementation is based on following papers:
	
> * Blei, D.M., Ng, A.Y., &Jordan, M.I. (2003).Latent dirichlet allocation.Journal of machine Learning research, 3(Jan), 993 - 1022.
> * Newman, D., Asuncion, A., Smyth, P., &Welling, M. (2009).Distributed algorithms for topic models.Journal of Machine Learning Research, 10(Aug), 1801 - 1828.
	
LDAModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, seed=?)

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
* `k` : 토픽의 개수, 1 ~ 32767의 정수.
* `alpha` : 문헌-토픽 디리클레 분포의 하이퍼 파라미터
* `eta` : 토픽-단어 디리클레 분포의 하이퍼 파라미터
* `seed` : 난수의 시드값. 기본값은 C++의 `std::random_device{}`이 생성하는 임의의 정수입니다.
이 값을 고정하더라도 `train`시 `workers`를 2 이상으로 두면, 멀티 스레딩 과정에서 발생하는 우연성 때문에 실행시마다 결과가 달라질 수 있습니다.
)"");

DOC_SIGNATURE_EN(LDA_add_doc__doc__,
	"add_doc(self, words)",
	u8R""(Add a new document into the model instance and return an index of the inserted document.

* `words` : an iterable of `str`
)"");

DOC_SIGNATURE_EN(LDA_make_doc__doc__,
	"make_doc(self, words)",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iterable of `str`
)"");

DOC_SIGNATURE_EN(LDA_train__doc__,
	"train(self, iter=10, workers=0)",
	u8R""(Train the model using Gibbs-sampling with `iter` iterations. Return `None`. 
After calling this method, you cannot `tomotopy.LDAModel.add_doc` more.

* `workers` : an integer indicating the number of workers to perform samplings. 
If `workers` is 0, the number of cores in the system will be used.
)"");

DOC_SIGNATURE_EN(LDA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer, indicating the topic, in range [0, `k`)
)"");

DOC_SIGNATURE_EN(LDA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in current topic.

* `topic_id` : an integer, indicating the topic, in range [0, `k`)
)"");

DOC_SIGNATURE_EN(LDA_get_count_by_topics__doc__,
	"get_count_by_topics(self)",
	u8R""(Return the number of words allocated to each topic.)"");

DOC_SIGNATURE_EN(LDA_infer__doc__,
	"infer(self, document, iter=100, tolerance=-1)",
	u8R""(Return the inferred topic distribution from unseen `document`s.
The return type is (a topic distribution of `document`, log_likelihood:`float`) or (a `list` of topic distribution of `document`, log_likelihood:`float`)

* `document` : an instance of `tomotopy.Document` or a `list` of instances of `tomotopy.Document` to be inferred by the model.
It can be acquired from `tomotopy.LDAModel.make_doc` method.
* `iter` : an integer indicating the number of iteration to estimate the distribution of topics of `document`.
The higher value will generate a more accuracy result.
* `tolerance` isn't currently used.
)"");

DOC_SIGNATURE_EN(LDA_save__doc__,
	"save(self, filename, full=True)",
	u8R""(Save the model instance to file `filename`. Return `None`.

If `full` is `True`, the model with its all documents and state will be saved. If you want to train more after, use full model.
If `False`, only topic paramters of the model will be saved. This model can be only used for inference of an unseen document.
)"");

DOC_SIGNATURE_EN(LDA_load__doc__,
	"load(filename)",
	u8R""(Return the model instance loaded from file `filename`.)"");

/*
	class DMR
*/
DOC_SIGNATURE_EN(DMR___init____doc__,
	"DMRModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=1e-10, seed=?)",
	u8R""(This type provides Dirichlet Multinomial Regression(DMR) topic model and its implementation is based on following papers:

> * Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.

DMRModel(tw=TermWeight.ONE, min_cf=0, k=1, alpha=0.1, eta=0.01, sigma=1.0, alpha_epsilon=1e-10, seed=?)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k` : the number of topics between 1 ~ 32767
* `alpha` : exponential of mean of normal distribution for `lambdas`
* `eta` : hyperparameter of Dirichlet distribution for topic - word
* `sigma` : standard deviation of normal distribution for `lambdas`
* `alpha_epsilon` : small value for preventing `exp(lambdas)` to be zero
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"");

DOC_SIGNATURE_EN(DMR_add_doc__doc__,
	"add_doc(self, words, metadata='')",
	u8R""(Add a new document into the model instance with `metadata` and return an index of the inserted document.

* `words` : an iterable of `str`
* `metadata` : a `str` indicating metadata of the document (e.g., author, title or year)
)"");

DOC_SIGNATURE_EN(DMR_make_doc__doc__,
	"make_doc(self, words, metadata='')",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` and `metadata` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iteratable of `str`
* `metadata` : a `str` indicating metadata of the document (e.g., author, title or year)
)"");

/*
	class HDP
*/
DOC_SIGNATURE_EN(HDP___init____doc__,
	"HDPModel(tw=TermWeight.ONE, min_cf=0, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=?)",
	u8R""(This type provides Hierarchical Dirichlet Process(HDP) topic model and its implementation is based on following papers:

> * Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
> * Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

HDPModel(tw=TermWeight.ONE, min_cf=0, initial_k=1, alpha=0.1, eta=0.01, gamma=0.1, seed=?)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `initial_k` : the initial number of topics between 1 ~ 32767.
The number of topics will be adjusted for data during training.
* `alpha` : concentration coeficient of Dirichlet Process for document-table 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `gamma` : concentration coeficient of Dirichlet Process for table-topic
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"");

DOC_SIGNATURE_EN(HDP_is_live_topic__doc__,
	"is_live_topic(self, topic_id)",
	u8R""(Return `True` if the topic `topic_id` is alive, otherwise return `False`.

* `topic_id` : an integer in range [0, `k`) indicating the topic
)"");

/*
	class MGLDA
*/
DOC_SIGNATURE_EN(MGLDA___init____doc__,
	"MGLDAModel(tw=TermWeight.ONE, min_cf=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=?)",
	u8R""(This type provides Multi Grain Latent Dirichlet Allocation(MG-LDA) topic model and its implementation is based on following papers:

> * Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

MGLDAModel(tw=TermWeight.ONE, min_cf=0, k_g=1, k_l=1, t=3, alpha_g=0.1, alpha_l=0.1, alpha_mg=0.1, alpha_ml=0.1, eta_g=0.01, eta_l=0.01, gamma=0.1, seed=?)

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
)"");

DOC_SIGNATURE_EN(MGLDA_add_doc__doc__,
	"add_doc(self, words, delimiter='.')",
	u8R""(Add a new document into the model instance and return an index of the inserted document.

* `words` : an iterable of `str`
* `delimiter` : a sentence separator. `words` will be separated by this value into sentences.
)"");

DOC_SIGNATURE_EN(MGLDA_make_doc__doc__,
	"make_doc(self, words, delimiter='.')",
	u8R""(Return a new `tomotopy.Document` instance for an unseen document with `words` that can be used for `tomotopy.LDAModel.infer` method.

* `words` : an iteratable of `str`
* `delimiter` : a sentence separator. `words` will be separated by this value into sentences.
)"");

DOC_SIGNATURE_EN(MGLDA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer. 
A number in range [0, `k_g`) indicates a global topic and 
a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
)"");

DOC_SIGNATURE_EN(MGLDA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in current topic.

* `topic_id` : an integer. 
A number in range [0, `k_g`) indicates a global topic and 
a number in range [`k_g`, `k_g` + `k_l`) indicates a local topic.
)"");

/*
	class PA
*/
DOC_SIGNATURE_EN(PA___init____doc__,
	"PAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)",
	u8R""(This type provides Pachinko Allocation(PA) topic model and its implementation is based on following papers:

> * Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.

PAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k1` : the number of super topics between 1 ~ 32767
* `k2` : the number of sub topics between 1 ~ 32767
* `alpha` : initial hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"");

DOC_SIGNATURE_EN(PA_get_sub_topic_dist__doc__,
	"get_sub_topic_dist(self, topic_id)",
	u8R""(Return a distribution of the sub topics in a super topic `topic_id`.)"");

/*
	class HPA
*/
DOC_SIGNATURE_EN(HPA___init____doc__,
	"HPAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)",
	u8R""(This type provides Hierarchical Pachinko Allocation(HPA) topic model and its implementation is based on following papers:

> * Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.

HPAModel(tw=TermWeight.ONE, min_cf=0, k1=1, k2=1, alpha=0.1, eta=0.01, seed=?)

* `tw` : term weighting scheme in `tomotopy.TermWeight`. The default value is TermWeight.ONE
* `min_cf` : minimum frequency of words. Words with a smaller collection frequency than `min_cf` are excluded from the model.
The default value is 0, which means no words are excluded.
* `k1` : the number of super topics between 1 ~ 32767
* `k2` : the number of sub topics between 1 ~ 32767
* `alpha` : initial hyperparameter of Dirichlet distribution for document-topic 
* `eta` : hyperparameter of Dirichlet distribution for topic-word
* `seed` : random seed. default value is a random number from `std::random_device{}` in C++
)"");

DOC_SIGNATURE_EN(HPA_get_topic_words__doc__,
	"get_topic_words(self, topic_id, top_n=10)",
	u8R""(Return the `top_n` words and its probability in the topic `topic_id`. 
The return type is a `list` of (word:`str`, probability:`float`).

* `topic_id` : an integer. 
0 indicates the top topic, 
a number in range [1, 1 + `k1`) indicates a super topic and
a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
)"");

DOC_SIGNATURE_EN(HPA_get_topic_word_dist__doc__,
	"get_topic_word_dist(self, topic_id)",
	u8R""(Return the word distribution of the topic `topic_id`.
The returned value is a `list` that has `len(vocabs)` fraction numbers indicating probabilities for each word in current topic.

* `topic_id` : an integer. 
0 indicates the top topic, 
a number in range [1, 1 + `k1`) indicates a super topic and
a number in range [1 + `k1`, 1 + `k1` + `k2`) indicates a sub topic.
)"");