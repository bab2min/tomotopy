#pragma once
#include <unordered_set>
#include <numeric>
#include "TopicModel.hpp"
#include <Eigen/Dense>
#include "../Utils/Utils.hpp"
#include "../Utils/math.h"
#include "../Utils/sample.hpp"

/*
Implementation of LDA using Gibbs sampling by bab2min

* Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.
* Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.

Term Weighting Scheme is based on following paper:
* Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.

*/

#define SWITCH_TW(TW, MDL, ...) do{ switch (TW)\
		{\
		case TermWeight::one:\
			return new MDL<TermWeight::one>(__VA_ARGS__);\
		case TermWeight::idf:\
			return new MDL<TermWeight::idf>(__VA_ARGS__);\
		case TermWeight::pmi:\
			return new MDL<TermWeight::pmi>(__VA_ARGS__);\
		}\
		return nullptr; } while(0)

#define GETTER(name, type, field) type get##name() const override { return field; }

namespace tomoto
{
	enum class TermWeight { one, idf, pmi, size };

	template<typename _Scalar, bool _Shared = false>
	struct ShareableVector : public Eigen::Matrix<_Scalar, -1, 1>
	{
		using Eigen::Matrix<_Scalar, -1, 1>::Matrix;
		void init(_Scalar* ptr, Eigen::Index len)
		{
			*this = Eigen::Matrix<_Scalar, -1, 1>::Zero(len);
		}
	};

	template<typename _Scalar>
	struct ShareableVector<_Scalar, true> : Eigen::Map<Eigen::Matrix<_Scalar, -1, 1>>
	{
		using Eigen::Map<Eigen::Matrix<_Scalar, -1, 1>>::Map;
		ShareableVector() : Eigen::Map<Eigen::Matrix<_Scalar, -1, 1>>{ nullptr, 0 }
		{}

		void init(_Scalar* ptr, Eigen::Index len)
		{
			// is this the best way??
			this->m_data = ptr;
			((Eigen::internal::variable_if_dynamic<Eigen::Index, -1>*)&this->m_rows)->setValue(len);
		}
	};

	template<TermWeight _TW, bool _Shared = false>
	struct DocumentLDA : public DocumentBase
	{
	public:
		using DocumentBase::DocumentBase;
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		tvector<TID> Zs;
		tvector<FLOAT> wordWeights;
		ShareableVector<WeightType, _Shared> numByTopic;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentBase, Zs, wordWeights);

		void update(WeightType* ptr, size_t K)
		{
			numByTopic.init(ptr, K);
			for (size_t i = 0; i < Zs.size(); ++i)
			{
				numByTopic[Zs[i]] += _TW != TermWeight::one ? wordWeights[i] : 1;
			}
		}

		template<TermWeight __TW>
		typename std::enable_if<__TW == TermWeight::one, int32_t>::type getSumWordWeight() const
		{
			return this->words.size();
		}

		template<TermWeight __TW>
		typename std::enable_if<__TW != TermWeight::one, FLOAT>::type getSumWordWeight() const
		{
			return std::accumulate(wordWeights.begin(), wordWeights.end(), 0.f);
		}
	};

	template<TermWeight _TW>
	struct ModelStateLDA
	{
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		Eigen::Matrix<FLOAT, -1, 1> zLikelihood;
		Eigen::Matrix<WeightType, -1, 1> numByTopic;
		Eigen::Matrix<WeightType, -1, -1> numByTopicWord;

		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	class ILDAModel : public ITopicModel
	{
	public:
		using DefaultDocType = DocumentLDA<TermWeight::one>;
		static ILDAModel* create(TermWeight _weight, size_t _K = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words) const = 0;

		virtual TermWeight getTermWeight() const = 0;
		virtual size_t getOptimInterval() const = 0;
		virtual void setOptimInterval(size_t) = 0;
		virtual std::vector<size_t> getCountByTopic() const = 0;
		virtual size_t getK() const = 0;
		virtual FLOAT getAlpha() const = 0;
		virtual FLOAT getEta() const = 0;
	};

	template<TermWeight _TW, bool _Shared = false,
		typename _Interface = ILDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentLDA<_TW, _Shared>,
		typename _ModelState = ModelStateLDA<_TW>>
	class LDAModel : public TopicModel<_Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LDAModel<_TW, _Shared>, _Derived>::type, 
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LDAModel, _Derived>::type;
		using BaseClass = TopicModel<_Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;

		static constexpr const char* TWID = _TW == TermWeight::one ? "one" : (_TW == TermWeight::idf ? "idf" : "pmi");
		static constexpr const char* TMID = "LDA";
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		std::vector<FLOAT> vocabWeight;
		std::vector<TID> sharedZs;
		std::vector<FLOAT> sharedWordWeights;
		FLOAT alpha;
		FLOAT eta;
		TID K;
		size_t optimInterval = 0;
		Eigen::Matrix<WeightType, -1, -1> numByTopicDoc;

		void optimizeHyperparameter(ThreadPool& pool, _ModelState* localData)
		{
		}

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t vid) const
		{
			const size_t V = this->dict.size();
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + alpha)
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * eta);

			sample::prefixSum(zLikelihood.data(), K);
			return &zLikelihood[0];
		}

		template<int INC>
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID tid) const
		{
			size_t V = this->dict.size();
			assert(tid < K);
			assert(vid < V);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
		}

		void sampleDocument(_DocType& doc, _ModelState& ld, RANDGEN& rgs) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w]);
				auto dist = static_cast<const DerivedClass*>(this)->getZLikelihoods(ld, doc, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + K, rgs);
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w]);
			}
		}

		void trainOne(ThreadPool& pool, _ModelState* localData, RANDGEN* rgs)
		{
			std::vector<std::future<void>> res;
			const size_t chStride = std::min(pool.getNumWorkers() * 8, this->docs.size());
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this, ch, chStride](size_t threadId)
				{
					forRandom((this->docs.size() - 1 - ch) / chStride + 1, this->rg(), [&, this](size_t id)
					{
						static_cast<DerivedClass*>(this)->sampleDocument(this->docs[id * chStride + ch],
							localData[threadId], rgs[threadId]);
					});
				}));
			}
			for (auto&& r : res) r.get();
			static_cast<DerivedClass*>(this)->updateGlobal(pool, localData);
			if (optimInterval && (this->iterated + 1) % optimInterval == 0)
			{
				static_cast<DerivedClass*>(this)->optimizeHyperparameter(pool, localData);
			}
		}

		void updateGlobal(ThreadPool& pool, _ModelState* localData)
		{
			std::vector<std::future<void>> res(pool.getNumWorkers());

			this->tState = this->globalState;
			this->globalState = localData[0];
			for (size_t i = 1; i < pool.getNumWorkers(); ++i)
			{
				this->globalState.numByTopic += localData[i].numByTopic - this->tState.numByTopic;
				this->globalState.numByTopicWord += localData[i].numByTopicWord - this->tState.numByTopicWord;
			}

			// make all count being positive
			if (_TW != TermWeight::one)
			{
				this->globalState.numByTopic = this->globalState.numByTopic.cwiseMax(0);
				this->globalState.numByTopicWord = this->globalState.numByTopicWord.cwiseMax(0);
			}

			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, this, i](size_t threadId)
				{
					localData[i] = this->globalState;
				});
			}
			for (auto&& r : res) r.get();
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			double ll = 0;
			// doc-topic distribution
			ll += (math::lgammaT(K*alpha) - math::lgammaT(alpha)*K) * std::distance(_first, _last);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll -= math::lgammaT(doc.template getSumWordWeight<_TW>() + K * alpha);
				for (TID k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alpha);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = 0;
			const size_t V = this->dict.size();
			// topic-word distribution
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K;
			for (TID k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (VID v = 0; v < V; ++v)
				{
					assert(ld.numByTopicWord(k, v) >= 0);
					ll += math::lgammaT(ld.numByTopicWord(k, v) + eta);
				}
			}
			return ll;
		}

		double getLL() const
		{
			return static_cast<const DerivedClass*>(this)->template getLLDocs<>(this->docs.begin(), this->docs.end())
				+ static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
		}

		void prepareShared()
		{
			std::vector<tvector<TID>*> srcs;
			srcs.reserve(this->docs.size());
			for (auto&& doc : this->docs)
			{
				srcs.emplace_back(&doc.Zs);
			}
			tvector<TID>::trade(sharedZs, srcs.begin(), srcs.end());
			if (_TW != TermWeight::one)
			{
				std::vector<tvector<FLOAT>*> srcs;
				srcs.reserve(this->docs.size());
				for (auto&& doc : this->docs)
				{
					srcs.emplace_back(&doc.wordWeights);
				}
				tvector<FLOAT>::trade(sharedWordWeights, srcs.begin(), srcs.end());
			}
		}
		
		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			doc.numByTopic.init(_Shared ? (WeightType*)numByTopicDoc.col(docId).data() : nullptr, K);
			doc.Zs = tvector<TID>(wordSize);
			if(_TW != TermWeight::one) doc.wordWeights.resize(wordSize, 1);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->dict.size();
			this->globalState.zLikelihood = Eigen::Matrix<FLOAT, -1, 1>::Zero(K);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
			}
			if(_Shared) numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(K, this->docs.size());
		}

		struct Generator
		{
			std::uniform_int_distribution<TID> theta;
		};

		Generator makeGeneratorForInit() const
		{
			return Generator{ std::uniform_int_distribution<TID>{0, (TID)(K - 1)} };
		}

		void updateStateWithDoc(Generator& g, _ModelState& ld, RANDGEN& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			z = g.theta(rgs);
			addWordTo<1>(ld, doc, i, w, z);
		}

		template<typename _Generator>
		void initializeDocState(_DocType& doc, size_t docId, _Generator& g, _ModelState& ld, RANDGEN& rgs) const
		{
			std::vector<uint32_t> tf(this->dict.size());
			static_cast<const DerivedClass*>(this)->prepareDoc(doc, docId, doc.words.size());
			if (_TW == TermWeight::pmi)
			{
				fill(tf.begin(), tf.end(), 0);
				for (auto& w : doc.words) ++tf[w];
			}

			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (_TW == TermWeight::idf)
				{
					doc.wordWeights[i] = vocabWeight[doc.words[i]];
				}
				else if (_TW == TermWeight::pmi)
				{
					doc.wordWeights[i] = std::max((FLOAT)log(tf[doc.words[i]] / vocabWeight[doc.words[i]] / doc.words.size()), (FLOAT)0);
				}
				static_cast<const DerivedClass*>(this)->updateStateWithDoc(g, ld, rgs, doc, i);
			}
		}

		std::vector<size_t> _getTopicsCount() const
		{
			std::vector<size_t> cnt(K);
			for (auto& doc : this->docs)
			{
				for (auto z : doc.Zs) ++cnt[z];
			}
			return cnt;
		}

		DEFINE_SERIALIZER(vocabWeight, alpha, eta, K);

	public:
		LDAModel(size_t _K = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RANDGEN& _rg = RANDGEN{ std::random_device{}() })
			: BaseClass(_rg), K(_K), alpha(_alpha), eta(_eta)
		{ }
		GETTER(K, size_t, K);
		GETTER(Alpha, FLOAT, alpha);
		GETTER(Eta, FLOAT, eta);
		GETTER(OptimInterval, size_t, optimInterval);

		TermWeight getTermWeight() const override
		{
			return _TW;
		}

		void setOptimInterval(size_t _optimInterval) override
		{
			optimInterval = _optimInterval;
		}

		size_t addDoc(const std::vector<std::string>& words) override
		{
			return this->_addDoc(this->_makeDoc(words));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words) const override
		{
			return std::make_unique<_DocType>(this->_makeDocWithinVocab(words));
		}

		void updateDocs()
		{
			size_t docId = 0;
			for (auto& doc : this->docs)
			{
				doc.update(_Shared ? numByTopicDoc.col(docId++).data() : nullptr, K);
			}
		}


		void prepare(bool initDocs = true)
		{
			static_cast<DerivedClass*>(this)->updateWeakArray();
			static_cast<DerivedClass*>(this)->initGlobalState(initDocs);

			const size_t V = this->dict.size();

			if (initDocs)
			{
				std::vector<uint32_t> df, cf, tf;
				uint32_t totCf;

				// calculate weighting
				if (_TW != TermWeight::one)
				{
					df.resize(V);
					cf.resize(V);
					tf.resize(V);
					for (auto& doc : this->docs)
					{
						for (auto& w : doc.words)
						{
							++cf[w];
						}

						for (auto w : std::unordered_set<VID>{ doc.words.begin(), doc.words.end() })
						{
							++df[w];
						}
					}
					totCf = accumulate(cf.begin(), cf.end(), 0);
				}
				if (_TW == TermWeight::idf)
				{
					vocabWeight.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeight[i] = log(this->docs.size() / (FLOAT)df[i]);
					}
				}
				else if (_TW == TermWeight::pmi)
				{
					vocabWeight.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeight[i] = cf[i] / (float)totCf;
					}
				}

				auto generator = static_cast<DerivedClass*>(this)->makeGeneratorForInit();
				for (auto& doc : this->docs)
				{
					initializeDocState(doc, &doc - &this->docs[0], generator, this->globalState, this->rg);
				}
			}
			else
			{
				static_cast<DerivedClass*>(this)->updateDocs();
			}
			static_cast<DerivedClass*>(this)->prepareShared();
		}

		std::vector<size_t> getCountByTopic() const override
		{
			return static_cast<const DerivedClass*>(this)->_getTopicsCount();
		}

		std::vector<FLOAT> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<FLOAT> ret(K);
			FLOAT sum = doc.template getSumWordWeight<_TW>() + K * alpha;
			transform(doc.numByTopic.data(), doc.numByTopic.data() + K, ret.begin(), [sum, this](size_t n)
			{
				return (n + alpha) / sum;
			});
			return ret;
		}

		std::vector<FLOAT> _getWidsByTopic(TID tid) const
		{
			assert(tid < K);
			const size_t V = this->dict.size();
			std::vector<FLOAT> ret(V);
			FLOAT sum = this->globalState.numByTopic[tid] + V * eta;
			auto r = this->globalState.numByTopicWord.row(tid);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + eta) / sum;
			}
			return ret;
		}

		FLOAT infer(_DocType& doc, size_t maxIter, FLOAT tolerance) const
		{
			auto generator = static_cast<const DerivedClass*>(this)->makeGeneratorForInit();
			// temporary state variable
			RANDGEN rgc{};
			auto tmpState = this->globalState;

			initializeDocState(doc, -1, generator, tmpState, rgc);
			for (size_t i = 0; i < maxIter; ++i)
			{
				static_cast<const DerivedClass*>(this)->sampleDocument(doc, tmpState, rgc);
			}

			FLOAT ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
			ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(&doc, &doc + 1);
			return ll;
		}

		FLOAT infer(const std::vector<_DocType*>& docs, size_t maxIter, FLOAT tolerance) const
		{
			auto generator = static_cast<const DerivedClass*>(this)->makeGeneratorForInit();
			// temporary state variable
			RANDGEN rgc{};
			auto tmpState = this->globalState;
			for (auto doc : docs)
			{
				initializeDocState(*doc, -1, generator, tmpState, rgc);
			}
			for (size_t i = 0; i < maxIter; ++i)
			{
				for (auto doc : docs) static_cast<const DerivedClass*>(this)->sampleDocument(*doc, tmpState, rgc);
			}

			FLOAT ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
			auto tx = [](auto x)->_DocType& { return *x; };
			ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(
				makeTransformIter(docs.begin(), tx),
				makeTransformIter(docs.end(), tx));
			return ll;
		}
	};

	ILDAModel* ILDAModel::create(TermWeight _weight, size_t _K, FLOAT _alpha, FLOAT _eta, const RANDGEN& _rg)
	{
		SWITCH_TW(_weight, LDAModel, _K, _alpha, _eta, _rg);
	}
}