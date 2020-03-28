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
		}\
		return nullptr; } while(0)

#define GETTER(name, type, field) type get##name() const override { return field; }

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateLDA
	{
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		Eigen::Matrix<Float, -1, 1> zLikelihood;
		Eigen::Matrix<WeightType, -1, 1> numByTopic; // Dim: (Topic, 1)
		Eigen::Matrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic, Vocabs)
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
		static constexpr char TWID[] = "one\0";
	};

	template<>
	struct TwId<TermWeight::idf>
	{
		static constexpr char TWID[] = "idf\0";
	};

	template<>
	struct TwId<TermWeight::pmi>
	{
		static constexpr char TWID[] = "pmi\0";
	};

	template<TermWeight _tw, size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = ILDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentLDA<_tw, _Flags>,
		typename _ModelState = ModelStateLDA<_tw>>
	class LDAModel : public TopicModel<_Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LDAModel<_tw, _Flags>, _Derived>::type, 
		_DocType, _ModelState>,
		protected TwId<_tw>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LDAModel, _Derived>::type;
		using BaseClass = TopicModel<_Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend EtaHelper<DerivedClass, true>;
		friend EtaHelper<DerivedClass, false>;

		static constexpr char TMID[] = "LDA\0";
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		enum { m_flags = _Flags };

		std::vector<Float> vocabWeights;
		std::vector<Tid> sharedZs;
		std::vector<Float> sharedWordWeights;
		Tid K;
		Float alpha, eta;
		Eigen::Matrix<Float, -1, 1> alphas;
		std::unordered_map<std::string, std::vector<Float>> etaByWord;
		Eigen::Matrix<Float, -1, -1> etaByTopicWord; // (K, V)
		Eigen::Matrix<Float, -1, 1> etaSumByTopic; // (K, )
		size_t optimInterval = 10, burnIn = 0;
		Eigen::Matrix<WeightType, -1, -1> numByTopicDoc;
		
		struct ExtraDocData
		{
			std::vector<Vid> vChunkOffset;
			Eigen::Matrix<uint32_t, -1, -1> chunkOffsetByDoc;
		};

		ExtraDocData eddTrain;

		template<typename _List>
		static Float calcDigammaSum(ThreadPool* pool, _List list, size_t len, Float alpha)
		{
			auto listExpr = Eigen::Matrix<Float, -1, 1>::NullaryExpr(len, list);
			auto dAlpha = math::digammaT(alpha);

			size_t suggested = (len + 127) / 128;
			if (suggested <= 1 || !pool)
			{
				return (math::digammaApprox(listExpr.array() + alpha) - dAlpha).sum();
			}
			if (suggested > pool->getNumWorkers()) suggested = pool->getNumWorkers();
			
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
		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
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

		template<int INC>
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, Vid vid, Tid tid) const
		{
			assert(tid < K);
			assert(vid < this->realV);
			constexpr bool DEC = INC < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
		}

		/*
		main sampling procedure
		*/
		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			size_t b = 0, e = doc.words.size();
			if (_ps == ParallelScheme::partition)
			{
				b = edd.chunkOffsetByDoc(partitionId, docId);
				e = edd.chunkOffsetByDoc(partitionId + 1, docId);
			}

			size_t vOffset = (_ps == ParallelScheme::partition && partitionId) ? edd.vChunkOffset[partitionId - 1] : 0;

			for (size_t w = b; w < e; ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w]);
				Float* dist;
				if (etaByTopicWord.size())
				{
					dist = static_cast<const DerivedClass*>(this)->template
						getZLikelihoods<true>(ld, doc, docId, doc.words[w] - vOffset);
				}
				else
				{
					dist = static_cast<const DerivedClass*>(this)->template
						getZLikelihoods<false>(ld, doc, docId, doc.words[w] - vOffset);
				}
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + K, rgs);
				addWordTo<1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w]);
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _DocIter, typename _ExtraDocData>
		void performSampling(ThreadPool& pool, _ModelState* localData, RandGen* rgs, std::vector<std::future<void>>& res,
			_DocIter docFirst, _DocIter docLast, const _ExtraDocData& edd) const
		{
			// single-threaded sampling
			if (_ps == ParallelScheme::none)
			{
				size_t docId = 0;
				for (auto doc = docFirst; doc != docLast; ++doc)
				{
					static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
						*doc, edd, docId++,
						*localData, *rgs, this->iterated, 0);
				}
			}
			// multi-threaded sampling on partition ad update into global
			else if (_ps == ParallelScheme::partition)
			{
				const size_t chStride = pool.getNumWorkers();
				for (size_t i = 0; i < chStride; ++i)
				{
					res = pool.enqueueToAll([&, i, chStride](size_t partitionId)
					{
						size_t didx = (i + partitionId) % chStride;
						forRandom(((size_t)std::distance(docFirst, docLast) + (chStride - 1) - didx) / chStride, rgs[partitionId](), [&](size_t id)
						{
							static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
								docFirst[id * chStride + didx], edd, id * chStride + didx,
								localData[partitionId], rgs[partitionId], this->iterated, partitionId);
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
						forRandom(((size_t)std::distance(docFirst, docLast) + (chStride - 1) - ch) / chStride, rgs[threadId](), [&](size_t id)
						{
							static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
								docFirst[id * chStride + ch], edd, id * chStride + ch,
								localData[threadId], rgs[threadId], this->iterated, 0);
						});
					}));
				}
				for (auto& r : res) r.get();
				res.clear();
			}
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

				localData[partitionId].numByTopicWord = globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b);
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
				return (this->realV + 3) / 4;
			}
			if (_ps == ParallelScheme::copy_merge)
			{
				return (this->docs.size() + 1) / 2;
			}
			return (size_t)-1;
		}

		template<ParallelScheme _ps>
		void trainOne(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			std::vector<std::future<void>> res;
			try
			{
				performSampling<_ps, false>(pool, localData, rgs, res, 
					this->docs.begin(), this->docs.end(), eddTrain);
				static_cast<DerivedClass*>(this)->updateGlobalInfo(pool, localData);
				static_cast<DerivedClass*>(this)->template mergeState<_ps>(pool, this->globalState, this->tState, localData, rgs, eddTrain);
				static_cast<DerivedClass*>(this)->template sampleGlobalLevel<>(&pool, localData, rgs, this->docs.begin(), this->docs.end());
				if (this->iterated >= this->burnIn && optimInterval && (this->iterated + 1) % optimInterval == 0)
				{
					static_cast<DerivedClass*>(this)->optimizeParameters(pool, localData, rgs);
				}
			}
			catch (const exception::TrainingError& e)
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
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, RandGen*, const _ExtraDocData& edd) const
		{
			std::vector<std::future<void>> res;

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
					globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
				}
				globalState.numByTopic = globalState.numByTopicWord.rowwise().sum();

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
				res = pool.enqueueToAll([&](size_t partitionId)
				{
					size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
						e = edd.vChunkOffset[partitionId];
					globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b) = localData[partitionId].numByTopicWord;
				});
				for (auto& r : res) r.get();
				res.clear();

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
				}
				globalState.numByTopic = globalState.numByTopicWord.rowwise().sum();

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
			auto lgammaEta = math::lgammaT(eta);
			ll += math::lgammaT(V*eta) * K;
			for (Tid k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
					if (!ld.numByTopicWord(k, v)) continue;
					ll += math::lgammaT(ld.numByTopicWord(k, v) + eta) - lgammaEta;
					assert(isfinite(ll));
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
		
		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			doc.numByTopic.init((m_flags & flags::continuous_doc_data) ? topicDocPtr : nullptr, K);
			doc.Zs = tvector<Tid>(wordSize);
			if(_tw != TermWeight::one) doc.wordWeights.resize(wordSize, 1);
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
				etaByTopicWord.col(id) = Eigen::Map<Eigen::Matrix<Float, -1, 1>>{ it.second.data(), (Eigen::Index)it.second.size() };
			}
			etaSumByTopic = etaByTopicWord.rowwise().sum();
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Eigen::Matrix<Float, -1, 1>::Zero(K);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
			}
			if(m_flags & flags::continuous_doc_data) numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(K, this->docs.size());
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
		void updateStateWithDoc(Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
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
		void initializeDocState(_DocType& doc, WeightType* topicDocPtr, _Generator& g, _ModelState& ld, RandGen& rgs) const
		{
			std::vector<uint32_t> tf(this->realV);
			static_cast<const DerivedClass*>(this)->prepareDoc(doc, topicDocPtr, doc.words.size());
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

		std::vector<Float> _getWidsByTopic(Tid tid) const
		{
			assert(tid < this->globalState.numByTopic.rows());
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
			decltype(static_cast<const DerivedClass*>(this)->makeGeneratorForInit(nullptr)) generator;
			if (!(m_flags & flags::generator_by_doc))
			{
				generator = static_cast<const DerivedClass*>(this)->makeGeneratorForInit(nullptr);
			}

			if (_Together)
			{
				numWorkers = std::min(numWorkers, this->maxThreads[(size_t)_ps]);
				ThreadPool pool{ numWorkers };
				// temporary state variable
				RandGen rgc{};
				auto tmpState = this->globalState, tState = this->globalState;
				for (auto d = docFirst; d != docLast; ++d)
				{
					initializeDocState<true>(*d, nullptr, generator, tmpState, rgc);
				}

				std::vector<decltype(tmpState)> localData((m_flags & flags::shared_state) ? 0 : pool.getNumWorkers(), tmpState);
				std::vector<RandGen> rgs;
				for (size_t i = 0; i < pool.getNumWorkers(); ++i) rgs.emplace_back(rgc());

				ExtraDocData edd;
				if (_ps == ParallelScheme::partition)
				{
					updatePartition(pool, tmpState, localData.data(), docFirst, docLast, edd);
				}

				for (size_t i = 0; i < maxIter; ++i)
				{
					std::vector<std::future<void>> res;
					performSampling<_ps, true>(pool,
						(m_flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), res,
						docFirst, docLast, edd);
					static_cast<const DerivedClass*>(this)->template mergeState<_ps>(pool, tmpState, tState, localData.data(), rgs.data(), edd);
					static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<>(
						&pool, (m_flags & flags::shared_state) ? &tmpState : localData.data(), rgs.data(), docFirst, docLast);
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
					RandGen rgc{};
					auto tmpState = this->globalState;
					initializeDocState<true>(*d, nullptr, generator, tmpState, rgc);
					for (size_t i = 0; i < maxIter; ++i)
					{
						static_cast<const DerivedClass*>(this)->template sampleDocument<ParallelScheme::none, true>(*d, edd, -1, tmpState, rgc, i);
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
				ThreadPool pool{ numWorkers, numWorkers * 8 };
				ExtraDocData edd;
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
							static_cast<const DerivedClass*>(this)->template sampleDocument<ParallelScheme::none, true>(*d, edd, -1, tmpState, rgc, i);
							static_cast<const DerivedClass*>(this)->template sampleGlobalLevel<>(
								nullptr, &tmpState, &rgc, &*d, &*d + 1);
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

		DEFINE_TAGGED_SERIALIZER_WITH_VERSION(1, 0x00010001, vocabWeights, alpha, alphas, eta, K, etaByWord);

		LDAModel(size_t _K = 1, Float _alpha = 0.1, Float _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_rg), K(_K), alpha(_alpha), eta(_eta)
		{ 
			if (_K == 0 || _K >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong K value (K = %zd)", _K));
			if (_alpha <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong alpha value (alpha = %f)", _alpha));
			if (_eta <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong eta value (eta = %f)", _eta));
			alphas = Eigen::Matrix<Float, -1, 1>::Constant(K, alpha);
		}

		GETTER(K, size_t, K);
		GETTER(Alpha, Float, alpha);
		GETTER(Eta, Float, eta);
		GETTER(OptimInterval, size_t, optimInterval);
		GETTER(BurnInIteration, size_t, burnIn);

		Float getAlpha(Tid k1) const override { return alphas[k1]; }

		TermWeight getTermWeight() const override
		{
			return _tw;
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
			return make_unique<_DocType>(as_mutable(this)->template _makeDoc<true>(words));
		}

		size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer) override
		{
			return this->_addDoc(this->template _makeRawDoc<false>(rawStr, tokenizer));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer) const override
		{
			return make_unique<_DocType>(as_mutable(this)->template _makeRawDoc<true>(rawStr, tokenizer));
		}

		size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len) override
		{
			return this->_addDoc(this->_makeRawDoc(rawStr, words, pos, len));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len) const override
		{
			return make_unique<_DocType>(this->_makeRawDoc(rawStr, words, pos, len));
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			if (priors.size() != K) THROW_ERROR_WITH_INFO(exception::InvalidArgument, "priors.size() must be equal to K.");
			for (auto p : priors)
			{
				if (p < 0) THROW_ERROR_WITH_INFO(exception::InvalidArgument, "priors must not be less than 0.");
			}
			this->dict.add(word);
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
				doc.template update<>((m_flags & flags::continuous_doc_data) ? numByTopicDoc.col(docId++).data() : nullptr, *static_cast<DerivedClass*>(this));
			}
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0) override
		{
			if (initDocs) this->removeStopwords(minWordCnt, minWordDf, removeTopN);
			static_cast<DerivedClass*>(this)->updateWeakArray();
			static_cast<DerivedClass*>(this)->initGlobalState(initDocs);
			static_cast<DerivedClass*>(this)->prepareWordPriors();

			const size_t V = this->realV;

			if (initDocs)
			{
				std::vector<uint32_t> df, cf, tf;
				uint32_t totCf;

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
					totCf = accumulate(this->vocabCf.begin(), this->vocabCf.end(), 0);
				}
				if (_tw == TermWeight::idf)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = log(this->docs.size() / (Float)df[i]);
					}
				}
				else if (_tw == TermWeight::pmi)
				{
					vocabWeights.resize(V);
					for (size_t i = 0; i < V; ++i)
					{
						vocabWeights[i] = this->vocabCf[i] / (float)totCf;
					}
				}

				decltype(static_cast<DerivedClass*>(this)->makeGeneratorForInit(nullptr)) generator;
				if(!(m_flags & flags::generator_by_doc)) generator = static_cast<DerivedClass*>(this)->makeGeneratorForInit(nullptr);
				for (auto& doc : this->docs)
				{
					initializeDocState<false>(doc, (m_flags & flags::continuous_doc_data) ? numByTopicDoc.col(&doc - &this->docs[0]).data() : nullptr, generator, this->globalState, this->rg);
				}
			}
			else
			{
				static_cast<DerivedClass*>(this)->updateDocs();
				for (auto& doc : this->docs) doc.updateSumWordWeight(this->realV);
			}
			static_cast<DerivedClass*>(this)->prepareShared();
			BaseClass::prepare(initDocs, minWordCnt, minWordDf, removeTopN);
		}

		std::vector<size_t> getCountByTopic() const override
		{
			return static_cast<const DerivedClass*>(this)->_getTopicsCount();
		}

		std::vector<Float> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<Float> ret(K);
			Eigen::Map<Eigen::Matrix<Float, -1, 1>> { ret.data(), K }.array() = 
				(doc.numByTopic.array().template cast<Float>() + alphas.array()) / (doc.getSumWordWeight() + alphas.sum());
			return ret;
		}

	};

	template<TermWeight _tw, size_t _Flags>
	template<typename _TopicModel>
	void DocumentLDA<_tw, _Flags>::update(WeightType* ptr, const _TopicModel& mdl)
	{
		numByTopic.init(ptr, mdl.getK());
		for (size_t i = 0; i < Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numByTopic[Zs[i]] += _tw != TermWeight::one ? wordWeights[i] : 1;
		}
	}
}