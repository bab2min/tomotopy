#pragma once
#include <unordered_set>
#include <numeric>
#include "TopicModel.hpp"
#include <Eigen/Dense>
#include "../Utils/Utils.hpp"
#include "../Utils/math.h"
#include "../Utils/sample.hpp"

/*
Implementation of LDA using Collapsed Variational Bayes zero-order estimation by bab2min

* Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), 993-1022.

Term Weighting Scheme is based on following paper:
* Wilson, A. T., & Chew, P. A. (2010, June). Term weighting schemes for latent dirichlet allocation. In human language technologies: The 2010 annual conference of the North American Chapter of the Association for Computational Linguistics (pp. 465-473). Association for Computational Linguistics.

*/

#define GETTER(name, type, field) type get##name() const override { return field; }
namespace tomoto
{
	struct DocumentLDACVB0 : public DocumentBase
	{
	public:
		using DocumentBase::DocumentBase;

		Eigen::MatrixXf Zs;
		Eigen::VectorXf numByTopic;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentBase, Zs);

		template<typename _TopicModel> void update(Float* ptr, const _TopicModel& mdl);

		int32_t getSumWordWeight() const
		{
			return this->words.size();
		}
	};

	struct ModelStateLDACVB0
	{
		Eigen::VectorXf zLikelihood;
		Eigen::VectorXf numByTopic;
		Eigen::MatrixXf numByTopicWord;

		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	class ILDACVB0Model : public ITopicModel
	{
	public:
		using DefaultDocType = DocumentLDACVB0;
		static ILDACVB0Model* create(size_t _K = 1, Float _alpha = 0.1, Float _eta = 0.01, size_t _rg = std::random_device{}());

		virtual size_t addDoc(const std::vector<std::string>& words) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words) const = 0;
		TermWeight getTermWeight() const { return TermWeight::one; };
		virtual size_t getOptimInterval() const = 0;
		virtual void setOptimInterval(size_t) = 0;
		virtual void setBurnInIteration(size_t) {}
		virtual std::vector<size_t> getCountByTopic() const = 0;
		virtual size_t getK() const = 0;
		virtual Float getAlpha() const = 0;
		virtual Float getEta() const = 0;

		virtual std::vector<Float> getWordPrior(const std::string& word) const { return {}; }
		virtual void setWordPrior(const std::string& word, const std::vector<Float>& priors) {}
	};

	template<typename _Interface = ILDACVB0Model,
		typename _Derived = void, 
		typename _DocType = DocumentLDACVB0,
		typename _ModelState = ModelStateLDACVB0>
	class LDACVB0Model : public TopicModel<0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LDACVB0Model<>, _Derived>::type, 
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LDACVB0Model, _Derived>::type;
		using BaseClass = TopicModel<0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;

		static constexpr const char TWID[] = "one\0";
		static constexpr const char TMID[] = "LDA\0";

		Float alpha;
		Vector alphas;
		Float eta;
		Tid K;
		size_t optimInterval = 50;

		template<typename _List>
		static Float calcDigammaSum(_List list, size_t len, Float alpha)
		{
			auto listExpr = Vector::NullaryExpr(len, list);
			auto dAlpha = math::digammaT(alpha);
			return (math::digammaApprox(listExpr.array() + alpha) - dAlpha).sum();
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData)
		{
			const auto K = this->K;
			for (size_t i = 0; i < 5; ++i)
			{
				Float denom = calcDigammaSum([&](size_t i) { return this->docs[i].getSumWordWeight(); }, this->docs.size(), alphas.sum());
				for (size_t k = 0; k < K; ++k)
				{
					Float nom = calcDigammaSum([&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), alphas(k));
					alphas(k) = std::max(nom / denom * alphas(k), 1e-5f);
				}
			}
		}

		const Eigen::VectorXf& getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + eta)
				/ (ld.numByTopic.array().template cast<Float>() + V * eta);
			zLikelihood /= zLikelihood.sum() + 1e-10;
			return zLikelihood;
		}

		template<int _Inc, typename _Vec>
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, _Vec tDist) const
		{
			assert(vid < this->realV);
			constexpr bool _dec = _Inc < 0;
			doc.numByTopic += _Inc * tDist;
			if (_dec) doc.numByTopic = doc.numByTopic.cwiseMax(0);
			ld.numByTopic += _Inc * tDist;
			if (_dec) ld.numByTopic = ld.numByTopic.cwiseMax(0);
			ld.numByTopicWord.col(vid) += _Inc * tDist;
			if (_dec) ld.numByTopicWord.col(vid) = ld.numByTopicWord.col(vid).cwiseMax(0);
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs.col(w));
				doc.Zs.col(w) = static_cast<const DerivedClass*>(this)->getZLikelihoods(ld, doc, docId, doc.words[w]);
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs.col(w));
			}
		}

		template<typename _DocIter, typename _ExtraDocData>
		void updatePartition(ThreadPool& pool, _ModelState* localData, _DocIter first, _DocIter last, _ExtraDocData& edd)
		{
		}

		template<ParallelScheme _ps>
		void trainOne(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			std::vector<std::future<void>> res;
			const size_t chStride = std::min(pool.getNumWorkers() * 8, this->docs.size());
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this, ch, chStride](size_t threadId)
				{
					forShuffled((this->docs.size() - 1 - ch) / chStride + 1, rgs[threadId](), [&, this](size_t id)
					{
						static_cast<DerivedClass*>(this)->template sampleDocument<ParallelScheme::copy_merge>(
							this->docs[id * chStride + ch], 0, id * chStride + ch,
							localData[threadId], rgs[threadId], this->globalStep);
					});
				}));
			}
			for (auto& r : res) r.get();
			static_cast<DerivedClass*>(this)->updateGlobalInfo(pool, localData);
			static_cast<DerivedClass*>(this)->mergeState(pool, this->globalState, this->tState, localData);
			if (this->globalStep >= 250 && optimInterval && (this->globalStep + 1) % optimInterval == 0)
			{
				static_cast<DerivedClass*>(this)->optimizeParameters(pool, localData);
			}
		}

		void updateGlobalInfo(ThreadPool& pool, _ModelState* localData)
		{
			std::vector<std::future<void>> res;

			this->globalState.numByTopic.setZero();
			this->globalState.numByTopicWord.setZero();
			for (auto& doc : this->docs)
			{
				doc.numByTopic = doc.Zs.rowwise().sum();
				this->globalState.numByTopic += doc.numByTopic;
				for (size_t i = 0; i < doc.words.size(); ++i)
				{
					this->globalState.numByTopicWord.col(doc.words[i]) += doc.Zs.col(i);
				}
			}

			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res.emplace_back(pool.enqueue([&, i](size_t threadId)
				{
					localData[i] = this->globalState;
				}));
			}
			for (auto& r : res) r.get();
		}

		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData) const
		{
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
				ll -= math::lgammaT(doc.getSumWordWeight() + K * alpha);
				for (Tid k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alpha);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = 0;
			const size_t V = this->realV;
			// topic-word distribution
			// it has the very-small-value problem
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K;
			for (Tid k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
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
		}
		
		void prepareDoc(_DocType& doc, Float* topicDocPtr, size_t wordSize) const
		{
			doc.numByTopic = Eigen::VectorXf::Zero(K);
			doc.Zs = Eigen::MatrixXf::Zero(K, wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Vector::Zero(K);
			if (initDocs)
			{
				this->globalState.numByTopic = Vector::Zero(K);
				this->globalState.numByTopicWord = Matrix::Zero(K, V);
			}
		}

		struct Generator
		{
			std::uniform_int_distribution<Tid> theta;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			return Generator{ std::uniform_int_distribution<Tid>{0, (Tid)(K - 1)} };
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			doc.Zs.col(i).setZero();
			doc.Zs(g.theta(rgs), i) = 1;
			addWordTo<1>(ld, doc, i, doc.words[i], doc.Zs.col(i));
		}

		template<bool _Infer, typename _Generator>
		void initializeDocState(_DocType& doc, Float* topicDocPtr, _Generator& g, _ModelState& ld, _RandGen& rgs) const
		{
			std::vector<uint32_t> tf(this->realV);
			static_cast<const DerivedClass*>(this)->prepareDoc(doc, topicDocPtr, doc.words.size());
			
			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				static_cast<const DerivedClass*>(this)->template updateStateWithDoc<_Infer>(g, ld, rgs, doc, i);
			}
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			Eigen::VectorXf cnt = Eigen::VectorXf::Zero(K);
			for (auto& doc : this->docs)
			{
				cnt += doc.Zs.rowwise().sum();
			}

			return { cnt.data(), cnt.data() + K };
		}

		template<ParallelScheme _ps>
		size_t estimateMaxThreads() const
		{
			if (_ps == ParallelScheme::partition)
			{
				return this->realV / 4;
			}
			if (_ps == ParallelScheme::copy_merge)
			{
				return this->docs.size() / 2;
			}
			return (size_t)-1;
		}

		DEFINE_SERIALIZER(alpha, eta, K);

	public:
		LDACVB0Model(size_t _K = 1, Float _alpha = 0.1, Float _eta = 0.01, size_t _rg = std::random_device{}())
			: BaseClass(_rg), K(_K), alpha(_alpha), eta(_eta)
		{ 
			alphas = Vector::Constant(K, alpha);
		}
		GETTER(K, size_t, K);
		GETTER(Alpha, Float, alpha);
		GETTER(Eta, Float, eta);
		GETTER(OptimInterval, size_t, optimInterval);

	
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
			return std::make_unique<_DocType>(as_mutable(this)->template _makeDoc<true>(words));
		}

		void updateDocs()
		{
			for (auto& doc : this->docs)
			{
				doc.template update<>(nullptr, *static_cast<DerivedClass*>(this));
			}
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0, bool updateStopwords = true) override
		{
			if (initDocs) this->removeStopwords(minWordCnt, minWordDf, removeTopN, updateStopwords);
			static_cast<DerivedClass*>(this)->updateWeakArray();
			static_cast<DerivedClass*>(this)->initGlobalState(initDocs);

			if (initDocs)
			{
				auto generator = static_cast<DerivedClass*>(this)->makeGeneratorForInit(nullptr);
				for (auto& doc : this->docs)
				{
					initializeDocState<false>(doc, nullptr, generator, this->globalState, this->rg);
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

		std::vector<Float> _getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<Float> ret(K);
			Float sum = doc.getSumWordWeight() + K * alpha;
			transform(doc.numByTopic.data(), doc.numByTopic.data() + K, ret.begin(), [sum, this](size_t n)
			{
				return (n + alpha) / sum;
			});
			return ret;
		}

		std::vector<Float> _getWidsByTopic(Tid tid, bool normalize = true) const
		{
			assert(tid < K);
			const size_t V = this->realV;
			std::vector<Float> ret(V);
			Float sum = this->globalState.numByTopic[tid] + V * eta;
			auto r = this->globalState.numByTopicWord.row(tid);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + eta) / sum;
			}
			return ret;
		}

		template<bool _Together, ParallelScheme _ps, typename _Iter>
		std::vector<double> _infer(_Iter docFirst, _Iter docLast, size_t maxIter, Float tolerance, size_t numWorkers) const
		{
			return {};
		}
	};

	template<typename _TopicModel>
	void DocumentLDACVB0::update(Float * ptr, const _TopicModel & mdl)
	{
		numByTopic = Eigen::VectorXf::Zero(mdl.getK());
		for (size_t i = 0; i < Zs.cols(); ++i)
		{
			numByTopic += Zs.col(i);
		}
	}

	inline ILDACVB0Model* ILDACVB0Model::create(size_t _K, Float _alpha, Float _eta, const _RandGen& _rg)
	{
		return new LDACVB0Model<>(_K, _alpha, _eta, _rg);
	}

}