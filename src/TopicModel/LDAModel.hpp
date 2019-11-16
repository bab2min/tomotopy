#pragma once
#include <unordered_set>
#include <numeric>
#include "TopicModel.hpp"
#include "../Utils/EigenAddonOps.hpp"
#include "../Utils/Utils.hpp"
#include "../Utils/math.h"
#include "../Utils/sample.hpp"
#include "LDA.h"

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
		case TermWeight::idf_one:\
			return new MDL<TermWeight::idf_one>(__VA_ARGS__);\
		}\
		return nullptr; } while(0)

#define GETTER(name, type, field) type get##name() const override { return field; }

namespace tomoto
{
	template<TermWeight _TW>
	struct ModelStateLDA
	{
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		Eigen::Matrix<FLOAT, -1, 1> zLikelihood;
		Eigen::Matrix<WeightType, -1, 1> numByTopic; // Dim: (Topic, 1)
		Eigen::Matrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic, Vocabs)

		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	template<TermWeight _TW, size_t _Flags = 0,
		typename _Interface = ILDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentLDA<_TW, _Flags>,
		typename _ModelState = ModelStateLDA<_TW>>
	class LDAModel : public TopicModel<_Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LDAModel<_TW, _Flags>, _Derived>::type, 
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LDAModel, _Derived>::type;
		using BaseClass = TopicModel<_Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;

		static constexpr const char* TWID = _TW == TermWeight::one ? "one" : (_TW == TermWeight::idf ? "idf" : "pmi");
		static constexpr const char* TMID = "LDA";
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		std::vector<FLOAT> vocabWeights;
		std::vector<TID> sharedZs;
		std::vector<FLOAT> sharedWordWeights;
		TID K;
		FLOAT alpha;
		Eigen::Matrix<FLOAT, -1, 1> alphas;
		FLOAT eta;
		size_t optimInterval = 10, burnIn = 0;
		Eigen::Matrix<WeightType, -1, -1> numByTopicDoc;

		template<typename _List>
		static FLOAT calcDigammaSum(_List list, size_t len, FLOAT alpha)
		{
			FLOAT ret = 0;
			auto dAlpha = math::digammaT(alpha);
			for (size_t i = 0; i < len; ++i)
			{
				ret += math::digammaT(list(i) + alpha) - dAlpha;
			}
			return ret;
		}

		/*
		function for optimizing hyperparameters
		*/
		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			const auto K = this->K;
			for (size_t i = 0; i < 10; ++i)
			{
				FLOAT denom = calcDigammaSum([&](size_t i) { return this->docs[i].getSumWordWeight(); }, this->docs.size(), alphas.sum());
				for (size_t k = 0; k < K; ++k)
				{
					FLOAT nom = calcDigammaSum([&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), alphas(k));
					alphas(k) = std::max(nom / denom * alphas(k), 1e-5f);
				}
			}
		}

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * eta);

			sample::prefixSum(zLikelihood.data(), K);
			return &zLikelihood[0];
		}

		template<int INC>
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID tid) const
		{
			assert(tid < K);
			assert(vid < this->realV);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
		}

		/*
		main sampling procedure
		*/
		void sampleDocument(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w]);
				auto dist = static_cast<const DerivedClass*>(this)->getZLikelihoods(ld, doc, docId, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + K, rgs);
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w]);
			}
		}

		/*
		reserved for model which needs second sampling
		*/
		void sampleDocument_2(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
		}

		template<typename _DocIter, typename _SamplingFunc>
		void performSampling(ThreadPool& pool, _ModelState* localData, RandGen* rgs, std::vector<std::future<void>>& res,
			_DocIter docFirst, _DocIter docLast, _SamplingFunc func) const
		{
			if ((_Flags & flags::shared_state))
			{
				size_t docId = 0;
				for (auto doc = docFirst; doc != docLast; ++doc)
				{
					(static_cast<const DerivedClass*>(this)->*func)(
						*doc, docId++,
						*localData, *rgs, this->iterated);
				}
			}
			else
			{
				const size_t chStride = std::min(pool.getNumWorkers() * 8, (size_t)std::distance(docFirst, docLast));
				for (size_t ch = 0; ch < chStride; ++ch)
				{
					res.emplace_back(pool.enqueue([&, this, ch, chStride](size_t threadId)
					{
						forRandom(((size_t)std::distance(docFirst, docLast) - 1 - ch) / chStride + 1, rgs[threadId](), [&, this](size_t id)
						{
							(static_cast<const DerivedClass*>(this)->*func)(
								docFirst[id * chStride + ch], id * chStride + ch,
								localData[threadId], rgs[threadId], this->iterated);
						});
					}));
				}
				for (auto&& r : res) r.get();
				res.clear();
			}
		}

		void trainOne(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			std::vector<std::future<void>> res;
			try
			{
				performSampling(pool, localData, rgs, res, 
					this->docs.begin(), this->docs.end(), &DerivedClass::sampleDocument);
				if(&DerivedClass::sampleDocument_2 != &LDAModel::sampleDocument_2) performSampling(pool, localData, rgs, res,
					this->docs.begin(), this->docs.end(), &DerivedClass::sampleDocument_2);
				static_cast<DerivedClass*>(this)->updateGlobalInfo(pool, localData);
				if (!(_Flags & flags::shared_state))
				{
					static_cast<DerivedClass*>(this)->mergeState(pool, this->globalState, this->tState, localData, rgs);
				}
				static_cast<DerivedClass*>(this)->template sampleGlobalLevel<>(&pool, localData, rgs, this->docs.begin(), this->docs.end());
				if (this->iterated >= this->burnIn && optimInterval && (this->iterated + 1) % optimInterval == 0)
				{
					static_cast<DerivedClass*>(this)->optimizeParameters(pool, localData, rgs);
				}
			}
			catch (const exception::TrainingError& e)
			{
				for (auto&& r : res) if(r.valid()) r.get();
				throw;
			}
		}

		/*
		updates global informations after sampling documents
		ex) update new global K at HDP model
		*/
		void updateGlobalInfo(ThreadPool& pool, _ModelState* localData)
		{
		}

		/*
		merges multithreaded document sampling result
		*/
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, RandGen*) const
		{
			std::vector<std::future<void>> res(pool.getNumWorkers());

			tState = globalState;
			globalState = localData[0];
			for (size_t i = 1; i < pool.getNumWorkers(); ++i)
			{
				globalState.numByTopic += localData[i].numByTopic - tState.numByTopic;
				globalState.numByTopicWord += localData[i].numByTopicWord - tState.numByTopicWord;
			}

			// make all count being positive
			if (_TW != TermWeight::one)
			{
				globalState.numByTopic = globalState.numByTopic.cwiseMax(0);
				globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
			}

			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, i](size_t threadId)
				{
					localData[i] = globalState;
				});
			}
			for (auto&& r : res) r.get();
		}

		/*
		performs sampling which needs global state modification
		ex) document pathing at hLDA model
		* if pool is nullptr, workers has been already pooled and cannot branch works more.
		*/
		template<typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, RandGen* rgs, _DocIter first, _DocIter last) const
		{
		}

		template<typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, RandGen* rgs, _DocIter first, _DocIter last)
		{
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			double ll = 0;
			// doc-topic distribution
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll -= math::lgammaT(doc.getSumWordWeight() + alphas.sum()) - math::lgammaT(alphas.sum());
				for (TID k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alphas[k]) - math::lgammaT(alphas[k]);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = 0;
			const size_t V = this->realV;
			// topic-word distribution
			auto lgammaEta = math::lgammaT(eta);
			ll += math::lgammaT(V*eta) * K;
			for (TID k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (VID v = 0; v < V; ++v)
				{
					if (!ld.numByTopicWord(k, v)) continue;
					ll += math::lgammaT(ld.numByTopicWord(k, v) + eta) - lgammaEta;
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
			auto txZs = [](_DocType& doc) { return &doc.Zs; };
			tvector<TID>::trade(sharedZs, 
				makeTransformIter(this->docs.begin(), txZs),
				makeTransformIter(this->docs.end(), txZs));
			if (_TW != TermWeight::one)
			{
				auto txWeights = [](_DocType& doc) { return &doc.wordWeights; };
				tvector<FLOAT>::trade(sharedWordWeights,
					makeTransformIter(this->docs.begin(), txWeights),
					makeTransformIter(this->docs.end(), txWeights));
			}
		}
		
		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			doc.numByTopic.init((_Flags & flags::continuous_doc_data) ? topicDocPtr : nullptr, K);
			doc.Zs = tvector<TID>(wordSize);
			if(_TW != TermWeight::one) doc.wordWeights.resize(wordSize, 1);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Eigen::Matrix<FLOAT, -1, 1>::Zero(K);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
			}
			if(_Flags & flags::continuous_doc_data) numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(K, this->docs.size());
		}

		struct Generator
		{
			std::uniform_int_distribution<TID> theta;
		};

		Generator makeGeneratorForInit() const
		{
			return Generator{ std::uniform_int_distribution<TID>{0, (TID)(K - 1)} };
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			z = g.theta(rgs);
			addWordTo<1>(ld, doc, i, w, z);
		}

		template<bool _Infer, typename _Generator>
		void initializeDocState(_DocType& doc, WeightType* topicDocPtr, _Generator& g, _ModelState& ld, RandGen& rgs) const
		{
			std::vector<uint32_t> tf(this->realV);
			static_cast<const DerivedClass*>(this)->prepareDoc(doc, topicDocPtr, doc.words.size());
			if (_TW == TermWeight::pmi)
			{
				std::fill(tf.begin(), tf.end(), 0);
				for (auto& w : doc.words) if(w < this->realV) ++tf[w];
			}

			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				if (_TW == TermWeight::idf)
				{
					doc.wordWeights[i] = vocabWeights[doc.words[i]];
				}
				if (_TW == TermWeight::idf_one)
				{
					doc.wordWeights[i] = (vocabWeights[doc.words[i]] + 1) / 2;
				}
				else if (_TW == TermWeight::pmi)
				{
					doc.wordWeights[i] = std::max((FLOAT)log(tf[doc.words[i]] / vocabWeights[doc.words[i]] / doc.words.size()), (FLOAT)0);
				}
				static_cast<const DerivedClass*>(this)->template updateStateWithDoc<_Infer>(g, ld, rgs, doc, i);
			}
			doc.updateSumWordWeight(this->realV);
		}

		std::vector<size_t> _getTopicsCount() const
		{
			std::vector<size_t> cnt(K);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Zs[i]];
				}
			}
			return cnt;
		}

		std::vector<FLOAT> _getWidsByTopic(TID tid) const
		{
			assert(tid < K);
			const size_t V = this->realV;
			std::vector<FLOAT> ret(V);
			FLOAT sum = this->globalState.numByTopic[tid] + V * eta;
			auto r = this->globalState.numByTopicWord.row(tid);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + eta) / sum;
			}
			return ret;
		}

		template<bool _Together, typename _Iter>
		std::vector<double> _infer(_Iter docFirst, _Iter docLast, size_t maxIter, FLOAT tolerance, size_t numWorkers) const
		{
			auto generator = static_cast<const DerivedClass*>(this)->makeGeneratorForInit();
			if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
			ThreadPool pool(numWorkers, numWorkers * 8);
			if (_Together)
			{
				// temporary state variable
				RandGen rgc{};
				auto tmpState = this->globalState, tState = this->globalState;
				for (auto d = docFirst; d != docLast; ++d)
				{
					initializeDocState<true>(*d, nullptr, generator, tmpState, rgc);
				}

				std::vector<decltype(tmpState)> localData((_Flags & flags::shared_state) ? 0 : pool.getNumWorkers(), tmpState);
				std::vector<RandGen> rgs;
				for (size_t i = 0; i < pool.getNumWorkers(); ++i) rgs.emplace_back(rgc());

				for (size_t i = 0; i < maxIter; ++i)
				{
					std::vector<std::future<void>> res;
					performSampling(pool, 
						(_Flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), res,
						docFirst, docLast, &DerivedClass::sampleDocument);
					if (&DerivedClass::sampleDocument_2 != &LDAModel::sampleDocument_2) performSampling(pool, 
						(_Flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), res,
						docFirst, docLast, &DerivedClass::sampleDocument_2);
					if(!(_Flags & flags::shared_state)) static_cast<const DerivedClass*>(this)->mergeState(pool, tmpState, tState, localData.data(), rgs.data());
					static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<>(
						&pool, (_Flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), docFirst, docLast);
				}
				double ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(docFirst, docLast);
				return { ll };
			}
			else if (_Flags & flags::shared_state)
			{
				std::vector<double> ret;
				const double gllRest = static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				for (auto d = docFirst; d != docLast; ++d)
				{
					RandGen rgc{};
					auto tmpState = this->globalState;
					initializeDocState<true>(*d, nullptr, generator, tmpState, rgc);
					for (size_t i = 0; i < maxIter; ++i)
					{
						static_cast<const DerivedClass*>(this)->sampleDocument(*d, -1, tmpState, rgc, i);
						static_cast<const DerivedClass*>(this)->sampleDocument_2(*d, -1, tmpState, rgc, i);
						static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<>(
							&pool, &tmpState, &rgc, &*d, &*d + 1);
					}
					double ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - gllRest;
					ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(&*d, &*d + 1);
					ret.emplace_back(ll);
				}
				return ret;
			}
			else
			{
				std::vector<std::future<double>> res;
				const double gllRest = static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				for (auto d = docFirst; d != docLast; ++d)
				{
					res.emplace_back(pool.enqueue([&, d](size_t threadId)
					{
						RandGen rgc{};
						auto tmpState = this->globalState;
						initializeDocState<true>(*d, nullptr, generator, tmpState, rgc);
						for (size_t i = 0; i < maxIter; ++i)
						{
							static_cast<const DerivedClass*>(this)->sampleDocument(*d, -1, tmpState, rgc, i);
							static_cast<const DerivedClass*>(this)->sampleDocument_2(*d, -1, tmpState, rgc, i);
							static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<>(
								nullptr, &tmpState, &rgc, &*d, &*d + 1);
						}
						double ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - gllRest;
						ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(&*d, &*d + 1);
						return ll;
					}));
				}
				std::vector<double> ret;
				for (auto&& r : res) ret.emplace_back(r.get());
				return ret;
			}
		}

		DEFINE_SERIALIZER(vocabWeights, alpha, alphas, eta, K);

	public:
		LDAModel(size_t _K = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_rg), K(_K), alpha(_alpha), eta(_eta)
		{ 
			if (_K == 0 || _K >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong K value (K = %zd)", _K));
			if (_alpha <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong alpha value (alpha = %f)", _alpha));
			if (_eta <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong eta value (eta = %f)", _eta));
			alphas = Eigen::Matrix<FLOAT, -1, 1>::Constant(K, alpha);
		}

		GETTER(K, size_t, K);
		GETTER(Alpha, FLOAT, alpha);
		GETTER(Eta, FLOAT, eta);
		GETTER(OptimInterval, size_t, optimInterval);
		GETTER(BurnInIteration, size_t, burnIn);

		FLOAT getAlpha(TID k1) const override { return alphas[k1]; }

		TermWeight getTermWeight() const override
		{
			return _TW;
		}

		void setOptimInterval(size_t _optimInterval) override
		{
			optimInterval = _optimInterval;
		}

		void setBurnInIteration(size_t iteration) override
		{
			burnIn = iteration;
		}

		size_t addDoc(const std::vector<std::string>& words) override
		{
			return this->_addDoc(this->_makeDoc(words));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words) const override
		{
			return make_unique<_DocType>(this->_makeDocWithinVocab(words));
		}

		void updateDocs()
		{
			size_t docId = 0;
			for (auto& doc : this->docs)
			{
				doc.template update<>((_Flags & flags::continuous_doc_data) ? numByTopicDoc.col(docId++).data() : nullptr, *static_cast<DerivedClass*>(this));
			}
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t removeTopN = 0)
		{
			if (initDocs) this->removeStopwords(minWordCnt, removeTopN);
			static_cast<DerivedClass*>(this)->updateWeakArray();
			static_cast<DerivedClass*>(this)->initGlobalState(initDocs);

			const size_t V = this->realV;

			if (initDocs)
			{
				std::vector<uint32_t> df, cf, tf;
				uint32_t totCf;

				// calculate weighting
				if (_TW != TermWeight::one)
				{
					df.resize(V);
					tf.resize(V);
					for (auto& doc : this->docs)
					{
						for (auto w : std::unordered_set<VID>{ doc.words.begin(), doc.words.end() })
						{
							if (w >= this->realV) continue;
							++df[w];
						}
					}
					totCf = accumulate(this->vocabFrequencies.begin(), this->vocabFrequencies.end(), 0);
				}
				if (_TW == TermWeight::idf || _TW == TermWeight::idf_one)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = log(this->docs.size() / (FLOAT)df[i]);
					}
				}
				else if (_TW == TermWeight::pmi)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = this->vocabFrequencies[i] / (float)totCf;
					}
				}

				auto generator = static_cast<DerivedClass*>(this)->makeGeneratorForInit();
				for (auto& doc : this->docs)
				{
					initializeDocState<false>(doc, (_Flags & flags::continuous_doc_data) ? numByTopicDoc.col(&doc - &this->docs[0]).data() : nullptr, generator, this->globalState, this->rg);
				}
			}
			else
			{
				static_cast<DerivedClass*>(this)->updateDocs();
				for (auto& doc : this->docs) doc.updateSumWordWeight(this->realV);
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
			Eigen::Map<Eigen::Matrix<FLOAT, -1, 1>> { ret.data(), K }.array() = 
				(doc.numByTopic.array().template cast<FLOAT>() + alphas.array()) / (doc.getSumWordWeight() + alphas.sum());
			return ret;
		}

	};

	template<TermWeight _TW, size_t _Flags>
	template<typename _TopicModel>
	void DocumentLDA<_TW, _Flags>::update(WeightType* ptr, const _TopicModel& mdl)
	{
		numByTopic.init(ptr, mdl.getK());
		for (size_t i = 0; i < Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numByTopic[Zs[i]] += _TW != TermWeight::one ? wordWeights[i] : 1;
		}
	}
}