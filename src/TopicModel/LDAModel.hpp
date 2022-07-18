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

#ifdef TMT_SCALAR_RNG
#define TMT_SWITCH_TW(TW, SRNG, MDL, ...) do{\
		{\
			switch (TW){\
			case TermWeight::one:\
				return new MDL<TermWeight::one, ScalarRandGen>(__VA_ARGS__);\
			case TermWeight::idf:\
				return new MDL<TermWeight::idf, ScalarRandGen>(__VA_ARGS__);\
			case TermWeight::pmi:\
				return new MDL<TermWeight::pmi, ScalarRandGen>(__VA_ARGS__);\
			}\
		}\
		return nullptr; } while(0)
#else
#define TMT_SWITCH_TW(TW, SRNG, MDL, ...) do{\
		{\
			switch (TW){\
			case TermWeight::one:\
				return new MDL<TermWeight::one, RandGen>(__VA_ARGS__);\
			case TermWeight::idf:\
				return new MDL<TermWeight::idf, RandGen>(__VA_ARGS__);\
			case TermWeight::pmi:\
				return new MDL<TermWeight::pmi, RandGen>(__VA_ARGS__);\
			}\
		}\
		return nullptr; } while(0)
#endif

#define GETTER(name, type, field) type get##name() const override { return field; }

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateLDA
	{
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		Vector zLikelihood;
		Eigen::Matrix<WeightType, -1, 1> numByTopic; // Dim: (Topic, 1)
		//Eigen::Matrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic, Vocabs)
		ShareableMatrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic, Vocabs)
		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	namespace flags
	{
		enum
		{
			generator_by_doc = end_flag_of_TopicModel,
			end_flag_of_LDAModel = generator_by_doc << 1,
		};
	}


	template<typename _Model, bool _asymEta>
	class EtaHelper
	{
		const _Model& _this;
	public:
		EtaHelper(const _Model& p) : _this(p) {}

		Float getEta(size_t vid) const
		{
			return _this.eta;
		}

		Float getEtaSum() const
		{
			return _this.eta * _this.realV;
		}
	};

	template<typename _Model>
	class EtaHelper<_Model, true>
	{
		const _Model& _this;
	public:
		EtaHelper(const _Model& p) : _this(p) {}

		auto getEta(size_t vid) const
			-> decltype(_this.etaByTopicWord.col(vid).array())
		{
			return _this.etaByTopicWord.col(vid).array();
		}

		auto getEtaSum() const
			-> decltype(_this.etaSumByTopic.array())
		{
			return _this.etaSumByTopic.array();
		}
	};

	template<TermWeight _tw>
	struct TwId;

	template<>
	struct TwId<TermWeight::one>
	{
		static constexpr auto twid()
		{
			return serializer::to_key("one\0");
		}
	};

	template<>
	struct TwId<TermWeight::idf>
	{
		static constexpr auto twid()
		{
			return serializer::to_key("idf\0");
		}
	};

	template<>
	struct TwId<TermWeight::pmi>
	{
		static constexpr auto twid()
		{
			return serializer::to_key("pmi\0");
		}
	};

	inline Float floorBit(Float x, int bitsUnderPoint = 8)
	{
		Float s = (1 << bitsUnderPoint);
		return floor(x * s) / s;
	}

	// to make HDP friend of LDA for HDPModel::converToLDA
	template<TermWeight _tw,
		typename _RandGen,
		typename _Interface,
		typename _Derived,
		typename _DocType,
		typename _ModelState
	>
	class HDPModel;

	template<TermWeight _tw, typename _RandGen,
		size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = ILDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>
	>
	class LDAModel : public TopicModel<_RandGen, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LDAModel<_tw, _RandGen, _Flags>, _Derived>::type,
		_DocType, _ModelState>,
		protected TwId<_tw>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LDAModel, _Derived>::type;
		using BaseClass = TopicModel<_RandGen, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend EtaHelper<DerivedClass, true>;
		friend EtaHelper<DerivedClass, false>;

		template<TermWeight,
			typename,
			typename,
			typename,
			typename,
			typename>
		friend class HDPModel;

		static constexpr auto tmid()
		{
			return serializer::to_key("LDA\0");
		}

		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		enum { m_flags = _Flags };

		std::vector<Float> vocabWeights;
		std::vector<Tid> sharedZs;
		std::vector<Float> sharedWordWeights;
		Tid K;
		Float alpha, eta;
		Vector alphas;
		std::unordered_map<std::string, std::vector<Float>> etaByWord;
		Matrix etaByTopicWord; // (K, V)
		Vector etaSumByTopic; // (K, )
		uint32_t optimInterval = 10, burnIn = 0;
		Eigen::Matrix<WeightType, -1, -1> numByTopicDoc;
		
		struct ExtraDocData
		{
			std::vector<Vid> vChunkOffset;
			Eigen::Matrix<size_t, -1, -1> chunkOffsetByDoc;
		};

		ExtraDocData eddTrain;

		template<typename _List>
		static Float calcDigammaSum(ThreadPool* pool, _List list, size_t len, Float alpha)
		{
			auto listExpr = Vector::NullaryExpr(len, list);
			auto dAlpha = math::digammaT(alpha);

			size_t suggested = (len + 127) / 128;
			if (pool && suggested > pool->getNumWorkers()) suggested = pool->getNumWorkers();
			if (suggested <= 1 || !pool)
			{
				return (math::digammaApprox(listExpr.array() + alpha) - dAlpha).sum();
			}

			
			std::vector<std::future<Float>> futures;
			for (size_t i = 0; i < suggested; ++i)
			{
				size_t start = (len * i / suggested + 15) & ~0xF,
					end = std::min((len * (i + 1) / suggested + 15) & ~0xF, len);
				futures.emplace_back(pool->enqueue([&, start, end, dAlpha](size_t)
				{
					return (math::digammaApprox(listExpr.array().segment(start, end - start) + alpha) - dAlpha).sum();
				}));
			}
			Float ret = 0;
			for (auto& f : futures) ret += f.get();
			return ret;
		}

		/*
			function for optimizing hyperparameters
		*/
		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			const auto K = this->K;
			for (size_t i = 0; i < 10; ++i)
			{
				Float denom = calcDigammaSum(&pool, [&](size_t i) { return this->docs[i].getSumWordWeight(); }, this->docs.size(), alphas.sum());
				for (size_t k = 0; k < K; ++k)
				{
					Float nom = calcDigammaSum(&pool, [&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), alphas(k));
					alphas(k) = std::max(nom / denom * alphas(k), 1e-5f);
				}
			}
		}

		template<bool _asymEta>
		EtaHelper<DerivedClass, _asymEta> getEtaHelper() const
		{
			return EtaHelper<DerivedClass, _asymEta>{ *static_cast<const DerivedClass*>(this) };
		}

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());
			sample::prefixSum(zLikelihood.data(), K);
			return &zLikelihood[0];
		}

		template<int _inc>
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, Tid tid) const
		{
			assert(tid < K);
			assert(vid < this->realV);
			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<_dec>(doc.numByTopic[tid], _inc * weight);
			updateCnt<_dec>(ld.numByTopic[tid], _inc * weight);
			updateCnt<_dec>(ld.numByTopicWord(tid, vid), _inc * weight);
		}

		void resetStatistics()
		{
			this->globalState.numByTopic.setZero();
			this->globalState.numByTopicWord.setZero();
			for (auto& doc : this->docs)
			{
				doc.numByTopic.setZero();
				for (size_t w = 0; w < doc.words.size(); ++w)
				{
					if (doc.words[w] >= this->realV) continue;
					addWordTo<1>(this->globalState, doc, w, doc.words[w], doc.Zs[w]);
				}
			}
		}

		/*
		called once before sampleDocument
		*/
		void presampleDocument(_DocType& doc, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt) const
		{
		}

		/*
		main sampling procedure (can be called one or more by ParallelScheme)
		*/
		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			size_t b = 0, e = doc.words.size();
			if (_ps == ParallelScheme::partition)
			{
				b = edd.chunkOffsetByDoc(partitionId, docId);
				e = edd.chunkOffsetByDoc(partitionId + 1, docId);
			}

			for (size_t w = b; w < e; ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				static_cast<const DerivedClass*>(this)->template addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w]);
				Float* dist;
				if (etaByTopicWord.size())
				{
					dist = static_cast<const DerivedClass*>(this)->template
						getZLikelihoods<true>(ld, doc, docId, doc.words[w]);
				}
				else
				{
					dist = static_cast<const DerivedClass*>(this)->template
						getZLikelihoods<false>(ld, doc, docId, doc.words[w]);
				}
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + K, rgs);
				static_cast<const DerivedClass*>(this)->template addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w]);
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _DocIter, typename _ExtraDocData>
		void performSampling(ThreadPool& pool, _ModelState* localData, _RandGen* rgs, std::vector<std::future<void>>& res,
			_DocIter docFirst, _DocIter docLast, const _ExtraDocData& edd) const
		{
			// single-threaded sampling
			if (_ps == ParallelScheme::none)
			{
				forShuffled((size_t)std::distance(docFirst, docLast), rgs[0](), [&](size_t id)
				{
					static_cast<const DerivedClass*>(this)->presampleDocument(docFirst[id], id, *localData, *rgs, this->globalStep);
					static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
						docFirst[id], edd, id,
						*localData, *rgs, this->globalStep, 0);

				});
			}
			// multi-threaded sampling on partition and update into global
			else if (_ps == ParallelScheme::partition)
			{
				const size_t chStride = pool.getNumWorkers();
				for (size_t i = 0; i < chStride; ++i)
				{
					res = pool.enqueueToAll([&, i, chStride](size_t partitionId)
					{
						size_t didx = (i + partitionId) % chStride;
						forShuffled(((size_t)std::distance(docFirst, docLast) + (chStride - 1) - didx) / chStride, rgs[partitionId](), [&](size_t id)
						{
							if (i == 0)
							{
								static_cast<const DerivedClass*>(this)->presampleDocument(
									docFirst[id * chStride + didx], id * chStride + didx,
									localData[partitionId], rgs[partitionId], this->globalStep
								);
							}
							static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
								docFirst[id * chStride + didx], edd, id * chStride + didx,
								localData[partitionId], rgs[partitionId], this->globalStep, partitionId
							);
						});
					});
					for (auto& r : res) r.get();
					res.clear();
				}
			}
			// multi-threaded sampling on copy and merge into global
			else if(_ps == ParallelScheme::copy_merge)
			{
				const size_t chStride = std::min(pool.getNumWorkers() * 8, (size_t)std::distance(docFirst, docLast));
				for (size_t ch = 0; ch < chStride; ++ch)
				{
					res.emplace_back(pool.enqueue([&, ch, chStride](size_t threadId)
					{
						forShuffled(((size_t)std::distance(docFirst, docLast) + (chStride - 1) - ch) / chStride, rgs[threadId](), [&](size_t id)
						{
							static_cast<const DerivedClass*>(this)->presampleDocument(
								docFirst[id * chStride + ch], id * chStride + ch, 
								localData[threadId], rgs[threadId], this->globalStep
							);
							static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
								docFirst[id * chStride + ch], edd, id * chStride + ch,
								localData[threadId], rgs[threadId], this->globalStep, 0
							);
						});
					}));
				}
				for (auto& r : res) r.get();
				res.clear();
			}
			else
			{
				throw std::runtime_error{ "Unsupported ParallelScheme" };
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _DocIter>
		void performSamplingGlobal(ThreadPool* pool, _ModelState& globalState, _RandGen* rgs, 
			_DocIter docFirst, _DocIter docLast) const
		{
		}

		template<typename _DocIter, typename _ExtraDocData>
		void updatePartition(ThreadPool& pool, const _ModelState& globalState, _ModelState* localData, _DocIter first, _DocIter last, _ExtraDocData& edd) const
		{
			size_t numPools = pool.getNumWorkers();
			if (edd.vChunkOffset.size() != numPools)
			{
				edd.vChunkOffset.clear();
				size_t totCnt = std::accumulate(this->vocabCf.begin(), this->vocabCf.begin() + this->realV, 0);
				size_t cumCnt = 0;
				for (size_t i = 0; i < this->realV; ++i)
				{
					cumCnt += this->vocabCf[i];
					if (cumCnt * numPools >= totCnt * (edd.vChunkOffset.size() + 1)) edd.vChunkOffset.emplace_back(i + 1);
				}

				edd.chunkOffsetByDoc.resize(numPools + 1, std::distance(first, last));
				size_t i = 0;
				for (; first != last; ++first, ++i)
				{
					auto& doc = *first;
					edd.chunkOffsetByDoc(0, i) = 0;
					size_t g = 0;
					for (size_t j = 0; j < doc.words.size(); ++j)
					{
						for (; g < numPools && doc.words[j] >= edd.vChunkOffset[g]; ++g)
						{
							edd.chunkOffsetByDoc(g + 1, i) = j;
						}
					}
					for (; g < numPools; ++g)
					{
						edd.chunkOffsetByDoc(g + 1, i) = doc.words.size();
					}
				}
			}
			static_cast<const DerivedClass*>(this)->distributePartition(pool, globalState, localData, edd);
		}

		template<typename _ExtraDocData>
		void distributePartition(ThreadPool& pool, const _ModelState& globalState, _ModelState* localData, const _ExtraDocData& edd) const
		{
			std::vector<std::future<void>> res = pool.enqueueToAll([&](size_t partitionId)
			{
				size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
					e = edd.vChunkOffset[partitionId];

				//localData[partitionId].numByTopicWord.matrix() = globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b);
				localData[partitionId].numByTopicWord.init((WeightType*)globalState.numByTopicWord.data(), globalState.numByTopicWord.rows(), globalState.numByTopicWord.cols());
				localData[partitionId].numByTopic = globalState.numByTopic;
				if (!localData[partitionId].zLikelihood.size()) localData[partitionId].zLikelihood = globalState.zLikelihood;
			});
			
			for (auto& r : res) r.get();
		}

		template<ParallelScheme _ps>
		size_t estimateMaxThreads() const
		{
			if (_ps == ParallelScheme::partition)
			{
				return std::max(((size_t)this->realV + 3) / 4, (size_t)1);
			}
			if (_ps == ParallelScheme::copy_merge)
			{
				return std::max((this->docs.size() + 1) / 2, (size_t)1);
			}
			return (size_t)-1;
		}

		template<ParallelScheme _ps>
		void trainOne(ThreadPool& pool, _ModelState* localData, _RandGen* rgs, bool freeze_topics = false)
		{
			std::vector<std::future<void>> res;
			try
			{
				static_cast<DerivedClass*>(this)->template performSampling<_ps, false>(pool, localData, rgs, res,
					this->docs.begin(), this->docs.end(), eddTrain
				);
				static_cast<DerivedClass*>(this)->updateGlobalInfo(pool, localData);
				static_cast<DerivedClass*>(this)->template mergeState<_ps>(pool, this->globalState, this->tState, localData, rgs, eddTrain);
				static_cast<DerivedClass*>(this)->template performSamplingGlobal<_ps, false>(&pool, this->globalState, rgs, 
					this->docs.begin(), this->docs.end()
				);
				
				if(freeze_topics) static_cast<DerivedClass*>(this)->template sampleGlobalLevel<GlobalSampler::freeze_topics>(
					&pool, &this->globalState, rgs, this->docs.begin(), this->docs.end()
				);
				else static_cast<DerivedClass*>(this)->template sampleGlobalLevel<GlobalSampler::train>(
					&pool, &this->globalState, rgs, this->docs.begin(), this->docs.end()
				);

				static_cast<DerivedClass*>(this)->template distributeMergedState<_ps>(pool, this->globalState, localData);
				
				if (this->globalStep >= this->burnIn && optimInterval && (this->globalStep + 1) % optimInterval == 0)
				{
					static_cast<DerivedClass*>(this)->optimizeParameters(pool, localData, rgs);
				}
			}
			catch (const exc::TrainingError&)
			{
				for (auto& r : res) if(r.valid()) r.get();
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
		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, _RandGen*, const _ExtraDocData& edd) const
		{
			if (_ps == ParallelScheme::copy_merge)
			{
				tState = globalState;
				globalState = localData[0];
				for (size_t i = 1; i < pool.getNumWorkers(); ++i)
				{
					globalState.numByTopicWord += localData[i].numByTopicWord - tState.numByTopicWord;
				}

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord.matrix() = globalState.numByTopicWord.cwiseMax(0);
				}
				globalState.numByTopic = globalState.numByTopicWord.rowwise().sum();
			}
			else if (_ps == ParallelScheme::partition)
			{
				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord.matrix() = globalState.numByTopicWord.cwiseMax(0);
				}
				globalState.numByTopic = globalState.numByTopicWord.rowwise().sum();
			}
		}

		template<ParallelScheme _ps>
		void distributeMergedState(ThreadPool& pool, _ModelState& globalState, _ModelState* localData) const
		{
			std::vector<std::future<void>> res;
			if (_ps == ParallelScheme::copy_merge)
			{
				for (size_t i = 0; i < pool.getNumWorkers(); ++i)
				{
					res.emplace_back(pool.enqueue([&, i](size_t)
					{
						localData[i] = globalState;
					}));
				}
			}
			else if (_ps == ParallelScheme::partition)
			{
				res = pool.enqueueToAll([&](size_t threadId)
				{
					localData[threadId].numByTopic = globalState.numByTopic;
				});
			}
			for (auto& r : res) r.get();
		}

		/*
		performs sampling which needs global state modification
		ex) document pathing at hLDA model
		* if pool is nullptr, workers has been already pooled and cannot branch works more.
		*/
		template<GlobalSampler _gs, typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, _RandGen* rgs, _DocIter first, _DocIter last) const
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
				for (Tid k = 0; k < K; ++k)
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
			if (etaByTopicWord.size())
			{
				for (Tid k = 0; k < K; ++k)
				{
					Float etasum = etaByTopicWord.row(k).sum();
					ll += math::lgammaT(etasum) - math::lgammaT(ld.numByTopic[k] + etasum);
					for (Vid v = 0; v < V; ++v)
					{
						if (!ld.numByTopicWord(k, v)) continue;
						ll += math::lgammaT(ld.numByTopicWord(k, v) + etaByTopicWord(k, v)) - math::lgammaT(etaByTopicWord(k, v));
						assert(std::isfinite(ll));
					}
				}
			}
			else
			{
				auto lgammaEta = math::lgammaT(eta);
				ll += math::lgammaT(V * eta) * K;
				for (Tid k = 0; k < K; ++k)
				{
					ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
					for (Vid v = 0; v < V; ++v)
					{
						if (!ld.numByTopicWord(k, v)) continue;
						ll += math::lgammaT(ld.numByTopicWord(k, v) + eta) - lgammaEta;
						assert(std::isfinite(ll));
					}
				}
			}
			return ll;
		}

		double getLL() const
		{
			return static_cast<const DerivedClass*>(this)->getLLDocs(this->docs.begin(), this->docs.end())
				+ static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
		}

		void prepareShared()
		{
			auto txZs = [](_DocType& doc) { return &doc.Zs; };
			tvector<Tid>::trade(sharedZs, 
				makeTransformIter(this->docs.begin(), txZs),
				makeTransformIter(this->docs.end(), txZs));
			if (_tw != TermWeight::one)
			{
				auto txWeights = [](_DocType& doc) { return &doc.wordWeights; };
				tvector<Float>::trade(sharedWordWeights,
					makeTransformIter(this->docs.begin(), txWeights),
					makeTransformIter(this->docs.end(), txWeights));
			}
		}

		void updateForCopy()
		{
			BaseClass::updateForCopy();
			size_t offset = 0;
			for (auto& doc : this->docs)
			{
				size_t size = doc.Zs.size();
				doc.Zs = tvector<Tid>{ sharedZs.data() + offset, size };
				if (_tw != TermWeight::one)
				{
					doc.wordWeights = tvector<Float>{ sharedWordWeights.data() + offset, size };
				}
				offset += size;
			}
		}
		
		WeightType* getTopicDocPtr(size_t docId) const
		{
			if (!(m_flags & flags::continuous_doc_data) || docId == (size_t)-1) return nullptr;
			return (WeightType*)numByTopicDoc.col(docId).data();
		}

		/*
		* called only when initializing a new doc, not when loading from saved model
		*/
		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			doc.numByTopic.init(getTopicDocPtr(docId), K, 1);
			doc.Zs = tvector<Tid>(wordSize, non_topic_id);
			if(_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		void prepareWordPriors()
		{
			if (etaByWord.empty()) return;
			etaByTopicWord.resize(K, this->realV);
			etaSumByTopic.resize(K);
			etaByTopicWord.array() = eta;
			for (auto& it : etaByWord)
			{
				auto id = this->dict.toWid(it.first);
				if (id == (Vid)-1 || id >= this->realV) continue;
				etaByTopicWord.col(id) = Eigen::Map<Vector>{ it.second.data(), (Eigen::Index)it.second.size() };
			}
			etaSumByTopic = etaByTopicWord.rowwise().sum();
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Vector::Zero(K);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				//this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
				this->globalState.numByTopicWord.init(nullptr, K, V);
			}
			if(m_flags & flags::continuous_doc_data) numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(K, this->docs.size());
		}

		struct Generator
		{
			Eigen::Rand::DiscreteGen<int32_t> theta;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			Generator g;
			g.theta = Eigen::Rand::DiscreteGen<int32_t>{ alphas.data(), alphas.data() + alphas.size() };
			return g;
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			if (etaByTopicWord.size())
			{
				auto col = etaByTopicWord.col(w);
				z = sample::sampleFromDiscrete(col.data(), col.data() + col.size(), rgs);
			}
			else
			{
				z = g.theta(rgs);
			}
			addWordTo<1>(ld, doc, i, w, z);
		}

		template<bool _Infer, typename _Generator>
		void initializeDocState(_DocType& doc, size_t docId, _Generator& g, _ModelState& ld, _RandGen& rgs) const
		{
			std::vector<uint32_t> tf(this->realV);
			static_cast<const DerivedClass*>(this)->prepareDoc(doc, docId, doc.words.size());
			_Generator g2;
			_Generator* selectedG = &g;
			if (m_flags & flags::generator_by_doc)
			{
				g2 = static_cast<const DerivedClass*>(this)->makeGeneratorForInit(&doc);
				selectedG = &g2;
			}
			if (_tw == TermWeight::pmi)
			{
				std::fill(tf.begin(), tf.end(), 0);
				for (auto& w : doc.words) if(w < this->realV) ++tf[w];
			}

			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				if (_tw == TermWeight::idf)
				{
					doc.wordWeights[i] = vocabWeights[doc.words[i]];
				}
				else if (_tw == TermWeight::pmi)
				{
					doc.wordWeights[i] = std::max((Float)log(tf[doc.words[i]] / vocabWeights[doc.words[i]] / doc.words.size()), (Float)0);
				}
				static_cast<const DerivedClass*>(this)->template updateStateWithDoc<_Infer>(*selectedG, ld, rgs, doc, i);
			}
			doc.updateSumWordWeight(this->realV);
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(K);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Zs[i]];
				}
			}
			return cnt;
		}

		std::vector<Float> _getWidsByTopic(size_t tid, bool normalize = true) const
		{
			assert(tid < this->globalState.numByTopic.rows());
			const size_t V = this->realV;
			std::vector<Float> ret(V);
			Float sum = this->globalState.numByTopic[tid] + V * eta;
			if (!normalize) sum = 1;
			auto r = this->globalState.numByTopicWord.row(tid);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + eta) / sum;
			}
			return ret;
		}

		template<bool together, ParallelScheme _ps, typename _Iter>
		std::vector<double> _infer(_Iter docFirst, _Iter docLast, size_t maxIter, Float tolerance, size_t numWorkers) const
		{
			decltype(static_cast<const DerivedClass*>(this)->makeGeneratorForInit(nullptr)) generator;
			if (!(m_flags & flags::generator_by_doc))
			{
				generator = static_cast<const DerivedClass*>(this)->makeGeneratorForInit(nullptr);
			}

			if (together)
			{
				numWorkers = std::min(numWorkers, this->maxThreads[(size_t)_ps]);
				ThreadPool pool{ numWorkers };
				// temporary state variable
				_RandGen rgc{};
				auto tmpState = this->globalState, tState = this->globalState;
				for (auto d = docFirst; d != docLast; ++d)
				{
					initializeDocState<true>(*d, -1, generator, tmpState, rgc);
				}

				std::vector<decltype(tmpState)> localData((m_flags & flags::shared_state) ? 0 : pool.getNumWorkers(), tmpState);
				std::vector<_RandGen> rgs;
				for (size_t i = 0; i < pool.getNumWorkers(); ++i) rgs.emplace_back(rgc());

				ExtraDocData edd;
				if (_ps == ParallelScheme::partition)
				{
					updatePartition(pool, tmpState, localData.data(), docFirst, docLast, edd);
				}

				for (size_t i = 0; i < maxIter; ++i)
				{
					std::vector<std::future<void>> res;
					static_cast<const DerivedClass*>(this)->template performSampling<_ps, true>(pool,
						(m_flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), res,
						docFirst, docLast, edd
					);
					static_cast<const DerivedClass*>(this)->template mergeState<_ps>(pool, tmpState, tState, localData.data(), rgs.data(), edd);
					static_cast<const DerivedClass*>(this)->template performSamplingGlobal<_ps, true>(&pool, tmpState, rgs.data(),
						docFirst, docLast
					);
					static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<GlobalSampler::inference>(
						&pool, (m_flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), docFirst, docLast
					);
					static_cast<const DerivedClass*>(this)->template distributeMergedState<_ps>(pool, tmpState, localData.data());
				}
				double ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(docFirst, docLast);
				return { ll };
			}
			else if (m_flags & flags::shared_state)
			{
				ThreadPool pool{ numWorkers };
				ExtraDocData edd;
				std::vector<double> ret;
				const double gllRest = static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				for (auto d = docFirst; d != docLast; ++d)
				{
					_RandGen rgc{};
					auto tmpState = this->globalState;
					initializeDocState<true>(*d, -1, generator, tmpState, rgc);
					for (size_t i = 0; i < maxIter; ++i)
					{
						static_cast<const DerivedClass*>(this)->presampleDocument(*d, -1, tmpState, rgc, i);
						static_cast<const DerivedClass*>(this)->template sampleDocument<ParallelScheme::none, true>(*d, edd, -1, tmpState, rgc, i);
						static_cast<const DerivedClass*>(this)->template performSamplingGlobal<_ps, true>(&pool, tmpState, &rgc,
							&*d, &*d + 1);
						static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<GlobalSampler::inference>(
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
				ThreadPool pool{ numWorkers, numWorkers * 8 };
				ExtraDocData edd;
				std::vector<std::future<double>> res;
				const double gllRest = static_cast<const DerivedClass*>(this)->getLLRest(this->globalState);
				for (auto d = docFirst; d != docLast; ++d)
				{
					res.emplace_back(pool.enqueue([&, d](size_t threadId)
					{
						_RandGen rgc{};
						auto tmpState = this->globalState;
						initializeDocState<true>(*d, -1, generator, tmpState, rgc);
						for (size_t i = 0; i < maxIter; ++i)
						{
							static_cast<const DerivedClass*>(this)->presampleDocument(*d, -1, tmpState, rgc, i);
							static_cast<const DerivedClass*>(this)->template sampleDocument<ParallelScheme::none, true>(
								*d, edd, -1, tmpState, rgc, i
							);
							static_cast<const DerivedClass*>(this)->template performSamplingGlobal<_ps, true>(nullptr, tmpState, &rgc,
								&*d, &*d + 1);
							static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<GlobalSampler::inference>(
								nullptr, &tmpState, &rgc, &*d, &*d + 1
							);
						}
						double ll = static_cast<const DerivedClass*>(this)->getLLRest(tmpState) - gllRest;
						ll += static_cast<const DerivedClass*>(this)->template getLLDocs<>(&*d, &*d + 1);
						return ll;
					}));
				}
				std::vector<double> ret;
				for (auto& r : res) ret.emplace_back(r.get());
				return ret;
			}
		}

	public:
		DEFINE_SERIALIZER_WITH_VERSION(0, vocabWeights, alpha, alphas, eta, K);

		DEFINE_TAGGED_SERIALIZER_WITH_VERSION(1, 0x00010001, vocabWeights, alpha, alphas, eta, K, etaByWord,
			burnIn, optimInterval);

		LDAModel(const LDAArgs& args, bool checkAlpha = true)
			: BaseClass(args.seed), K(args.k), alpha(args.alpha[0]), eta(args.eta)
		{
			if (K == 0 || K >= 0x80000000) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong K value (K = %zd)", K));

			if (args.alpha.size() == 1)
			{
				alphas = Vector::Constant(K, alpha);
			}
			else if (args.alpha.size() == args.k)
			{
				alphas = Eigen::Map<const Vector>(args.alpha.data(), args.alpha.size());
			}
			else if (checkAlpha)
			{
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong alpha value (len = %zd)", args.alpha.size()));
			}

			if ((alphas.array() <= 0).any()) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "wrong alpha value");
			if (eta <= 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong eta value (eta = %f)", eta));
		}

		GETTER(K, size_t, K);
		GETTER(Alpha, Float, alpha);
		GETTER(Eta, Float, eta);
		GETTER(OptimInterval, size_t, optimInterval);
		GETTER(BurnInIteration, size_t, burnIn);

		Float getAlpha(size_t k1) const override { return alphas[k1]; }

		TermWeight getTermWeight() const override
		{
			return _tw;
		}

		void setOptimInterval(size_t _optimInterval) override
		{
			if (_optimInterval > 0x7FFFFFFF) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "wrong value");
			optimInterval = (uint32_t)_optimInterval;
		}

		void setBurnInIteration(size_t iteration) override
		{
			if (iteration > 0x7FFFFFFF) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "wrong value");
			burnIn = (uint32_t)iteration;
		}

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) override
		{
			return this->_addDoc(this->template _makeFromRawDoc<false>(rawDoc, tokenizer));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const override
		{
			return std::make_unique<_DocType>(as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer));
		}

		size_t addDoc(const RawDoc& rawDoc) override
		{
			return this->_addDoc(this->_makeFromRawDoc(rawDoc));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const override
		{
			return std::make_unique<_DocType>(as_mutable(this)->template _makeFromRawDoc<true>(rawDoc));
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			if (priors.size() != K) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors.size() must be equal to K.");
			for (auto p : priors)
			{
				if (p < 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors must not be less than 0.");
			}
			this->dict.add(word);
			if (this->dict.size() > this->vocabCf.size())
			{
				this->vocabCf.resize(this->dict.size());
				this->vocabDf.resize(this->dict.size());
			}
			etaByWord.emplace(word, priors);
		}

		std::vector<Float> getWordPrior(const std::string& word) const override
		{
			if (etaByTopicWord.size())
			{
				auto id = this->dict.toWid(word);
				if (id == (Vid)-1) return {};
				auto col = etaByTopicWord.col(id);
				return std::vector<Float>{ col.data(), col.data() + col.size() };
			}
			else
			{
				auto it = etaByWord.find(word);
				if (it == etaByWord.end()) return {};
				return it->second;
			}
		}

		void updateDocs()
		{
			size_t docId = 0;
			for (auto& doc : this->docs)
			{
				doc.template update<>(getTopicDocPtr(docId++), *static_cast<DerivedClass*>(this));
			}
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0, bool updateStopwords = true) override
		{
			if (initDocs && updateStopwords) this->removeStopwords(minWordCnt, minWordDf, removeTopN);
			static_cast<DerivedClass*>(this)->updateWeakArray();
			static_cast<DerivedClass*>(this)->initGlobalState(initDocs);
			static_cast<DerivedClass*>(this)->prepareWordPriors();

			const size_t V = this->realV;
			if (V == 0) 
			{
				std::cerr << "[warn] No valid vocabs in the model!" << std::endl;
			}

			if (initDocs)
			{
				std::vector<uint32_t> df, cf, tf;
				size_t totCf;

				// calculate weighting
				if (_tw != TermWeight::one)
				{
					df.resize(V);
					tf.resize(V);
					for (auto& doc : this->docs)
					{
						for (auto w : std::unordered_set<Vid>{ doc.words.begin(), doc.words.end() })
						{
							if (w >= this->realV) continue;
							++df[w];
						}
					}
					totCf = std::accumulate(this->vocabCf.begin(), this->vocabCf.end(), 0);
				}
				if (_tw == TermWeight::idf)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = (Float)log(this->docs.size() / (double)df[i]);
					}
				}
				else if (_tw == TermWeight::pmi)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = (Float)(this->vocabCf[i] / (double)totCf);
					}
				}

				decltype(static_cast<DerivedClass*>(this)->makeGeneratorForInit(nullptr)) generator;
				if(!(m_flags & flags::generator_by_doc)) generator = static_cast<DerivedClass*>(this)->makeGeneratorForInit(nullptr);
				for (auto& doc : this->docs)
				{
					initializeDocState<false>(doc, &doc - &this->docs[0], generator, this->globalState, this->rg);
				}
			}
			else
			{
				static_cast<DerivedClass*>(this)->updateDocs();
				for (auto& doc : this->docs) doc.updateSumWordWeight(this->realV);
			}
			static_cast<DerivedClass*>(this)->prepareShared();
			BaseClass::prepare(initDocs, minWordCnt, minWordDf, removeTopN, updateStopwords);
		}

		std::vector<uint64_t> getCountByTopic() const override
		{
			return static_cast<const DerivedClass*>(this)->_getTopicsCount();
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), K };
			if (normalize)
			{
				m = (doc.numByTopic.array().template cast<Float>() + alphas.array()) / (doc.getSumWordWeight() + alphas.sum());
			}
			else
			{
				m = doc.numByTopic.array().template cast<Float>() + alphas.array();
			}
			return ret;
		}

	};

	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentLDA<_tw>::update(WeightType* ptr, const _TopicModel& mdl)
	{
		numByTopic.init(ptr, mdl.getK(), 1);
		for (size_t i = 0; i < Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numByTopic[Zs[i]] += _tw != TermWeight::one ? wordWeights[i] : 1;
		}
	}
}