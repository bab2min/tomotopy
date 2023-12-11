#pragma once
#include "LDAModel.hpp"
#include "PA.h"

/*
Implementation of Pachinko Allocation using Gibbs sampling by bab2min

Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.
*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStatePA : public ModelStateLDA<_tw>
	{
		using WeightType = typename ModelStateLDA<_tw>::WeightType;
		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;
		Eigen::Matrix<WeightType, -1, 1> numByTopic2;
		Vector subTmp;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_tw>, numByTopic1_2, numByTopic2);
	};

	template<TermWeight _tw, typename _RandGen,
		typename _Interface = IPAModel,
		typename _Derived = void,
		typename _DocType = DocumentPA<_tw>,
		typename _ModelState = ModelStatePA<_tw>>
	class PAModel : public LDAModel<_tw, _RandGen, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Tid K2;
		Float epsilon = 1e-5;
		size_t iteration = 5;

		Vector subAlphaSum; // len = K
		Matrix subAlphas; // len = K * K2
		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			const auto K = this->K;
			std::vector<std::future<void>> res;
			for (size_t k = 0; k < K; ++k)
			{
				res.emplace_back(pool.enqueue([&, k](size_t)
				{
					for (size_t i = 0; i < iteration; ++i)
					{
						Float denom = this->template calcDigammaSum<>(nullptr, [&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), subAlphaSum[k]);
						for (size_t k2 = 0; k2 < K2; ++k2)
						{
							Float nom = this->template calcDigammaSum<>(nullptr, [&](size_t i) { return this->docs[i].numByTopic1_2(k, k2); }, this->docs.size(), subAlphas(k, k2));
							subAlphas(k, k2) = std::max(nom / denom * subAlphas(k, k2), epsilon);
						}
						subAlphaSum[k] = subAlphas.row(k).sum();
					}
				}));
			}
			for (auto& r : res) r.get();
		}

		// topic 1 & 2 assignment likelihoods for new word. ret K*K2 FLOATs
		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			const auto eta = this->eta;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			
			ld.subTmp = (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic2.array().template cast<Float>() + etaHelper.getEtaSum());

			for (size_t k = 0; k < this->K; ++k)
			{
				zLikelihood.segment(K2 * k, K2) = (doc.numByTopic[k] + this->alpha)
					* (doc.numByTopic1_2.row(k).transpose().array().template cast<Float>() + subAlphas.row(k).transpose().array()) / (doc.numByTopic[k] + subAlphaSum[k])
					* ld.subTmp.array();
			}
			sample::prefixSum(zLikelihood.data(), zLikelihood.size());
			return &zLikelihood[0];
		}

		template<int _inc> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, Tid z1, Tid z2) const
		{
			assert(vid < this->realV);
			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<_dec>(doc.numByTopic[z1], _inc * weight);
			updateCnt<_dec>(doc.numByTopic1_2(z1, z2), _inc * weight);
			updateCnt<_dec>(ld.numByTopic[z1], _inc * weight);
			updateCnt<_dec>(ld.numByTopic2[z2], _inc * weight);
			updateCnt<_dec>(ld.numByTopic1_2(z1, z2), _inc * weight);
			updateCnt<_dec>(ld.numByTopicWord(z2, vid), _inc * weight);
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
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
				addWordTo<-1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w], doc.Z2s[w]);
				Float* dist;
				if (this->etaByTopicWord.size())
				{
					dist = getZLikelihoods<true>(ld, doc, docId, doc.words[w] - vOffset);
				}
				else
				{
					dist = getZLikelihoods<false>(ld, doc, docId, doc.words[w] - vOffset);
				}
				auto z = sample::sampleFromDiscreteAcc(dist, dist + this->K * K2, rgs);
				doc.Zs[w] = z / K2;
				doc.Z2s[w] = z % K2;
				addWordTo<1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w], doc.Z2s[w]);
			}
		}

		template<typename _ExtraDocData>
		void distributePartition(ThreadPool& pool, const _ModelState& globalState, _ModelState* localData, const _ExtraDocData& edd) const
		{
			std::vector<std::future<void>> res = pool.enqueueToAll([&](size_t partitionId)
			{
				size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
					e = edd.vChunkOffset[partitionId];

				localData[partitionId].numByTopicWord.matrix() = globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b);
				localData[partitionId].numByTopic = globalState.numByTopic;
				localData[partitionId].numByTopic1_2 = globalState.numByTopic1_2;
				localData[partitionId].numByTopic2 = globalState.numByTopic2;
				if (!localData[partitionId].zLikelihood.size()) localData[partitionId].zLikelihood = globalState.zLikelihood;
			});

			for (auto& r : res) r.get();
		}

		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, _RandGen*, const _ExtraDocData& edd) const
		{
			if (_ps == ParallelScheme::copy_merge)
			{
				tState = globalState;
				globalState = localData[0];
				for (size_t i = 1; i < pool.getNumWorkers(); ++i)
				{
					globalState.numByTopic += localData[i].numByTopic - tState.numByTopic;
					globalState.numByTopic1_2 += localData[i].numByTopic1_2 - tState.numByTopic1_2;
					globalState.numByTopic2 += localData[i].numByTopic2 - tState.numByTopic2;
					globalState.numByTopicWord += localData[i].numByTopicWord - tState.numByTopicWord;
				}

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopic = globalState.numByTopic.cwiseMax(0);
					globalState.numByTopic1_2 = globalState.numByTopic1_2.cwiseMax(0);
					globalState.numByTopic2 = globalState.numByTopic2.cwiseMax(0);
					globalState.numByTopicWord.matrix() = globalState.numByTopicWord.cwiseMax(0);
				}
			}
			else if (_ps == ParallelScheme::partition)
			{
				std::vector<std::future<void>> res;
				res = pool.enqueueToAll([&](size_t partitionId)
				{
					size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
						e = edd.vChunkOffset[partitionId];
					globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b) = localData[partitionId].numByTopicWord;
				});
				for (auto& r : res) r.get();

				tState.numByTopic1_2 = globalState.numByTopic1_2;
				globalState.numByTopic1_2 = localData[0].numByTopic1_2;
				for (size_t i = 1; i < pool.getNumWorkers(); ++i)
				{
					globalState.numByTopic1_2 += localData[i].numByTopic1_2 - tState.numByTopic1_2;
				}

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord.matrix() = globalState.numByTopicWord.cwiseMax(0);
				}
				globalState.numByTopic = globalState.numByTopic1_2.rowwise().sum();
				globalState.numByTopic2 = globalState.numByTopicWord.rowwise().sum();

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
					localData[threadId].numByTopic1_2 = globalState.numByTopic1_2;
					localData[threadId].numByTopic2 = globalState.numByTopic2;
				});
			}
			for (auto& r : res) r.get();
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			const auto alpha = this->alpha;
			float ll = (math::lgammaT(K*alpha) - math::lgammaT(alpha)*K) * std::distance(_first, _last);
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
			const size_t V = this->realV;
			const auto K = this->K;
			const auto eta = this->eta;
		
			double ll = 0;
			for (Tid k = 0; k < K; ++k)
			{
				ll += math::lgammaT(subAlphaSum[k]);
				ll -= math::lgammaT(ld.numByTopic[k] + subAlphaSum[k]);
				for (Tid k2 = 0; k2 < K2; ++k2)
				{
					ll -= math::lgammaT(subAlphas(k, k2));
					ll += math::lgammaT(ld.numByTopic1_2(k, k2) + subAlphas(k, k2));
				}
			}
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K2;
			for (Tid k2 = 0; k2 < K2; ++k2)
			{
				ll -= math::lgammaT(ld.numByTopic2[k2] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
					ll += math::lgammaT(ld.numByTopicWord(k2, v) + eta);
				}
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);

			doc.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
			doc.Z2s = tvector<Tid>(wordSize, non_topic_id);
		}

		void prepareWordPriors()
		{
			if (this->etaByWord.empty()) return;
			this->etaByTopicWord.resize(K2, this->realV);
			this->etaSumByTopic.resize(K2);
			this->etaByTopicWord.array() = this->eta;
			for (auto& it : this->etaByWord)
			{
				auto id = this->dict.toWid(it.first);
				if (id == (Vid)-1 || id >= this->realV) continue;
				this->etaByTopicWord.col(id) = Eigen::Map<Vector>{ it.second.data(), (Eigen::Index)it.second.size() };
			}
			this->etaSumByTopic = this->etaByTopicWord.rowwise().sum();
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Vector::Zero(this->K * K2);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
				this->globalState.numByTopic2 = Eigen::Matrix<WeightType, -1, 1>::Zero(K2);
				this->globalState.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
				//this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K2, V);
				this->globalState.numByTopicWord.init(nullptr, K2, V);
			}
		}

		struct Generator
		{
			std::uniform_int_distribution<Tid> theta, theta2;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			return Generator{ 
				std::uniform_int_distribution<Tid>{0, (Tid)(this->K - 1)},
				std::uniform_int_distribution<Tid>{0, (Tid)(K2 - 1)},
			};
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto w = doc.words[i];
			doc.Zs[i] = g.theta(rgs);
			if (this->etaByTopicWord.size())
			{
				auto col = this->etaByTopicWord.col(w);
				doc.Z2s[i] = sample::sampleFromDiscrete(col.data(), col.data() + col.size(), rgs);
			}
			else
			{
				doc.Z2s[i] = g.theta2(rgs);
			}
			addWordTo<1>(ld, doc, i, w, doc.Zs[i], doc.Z2s[i]);
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(this->K2);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Z2s.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Z2s[i]];
				}
			}
			return cnt;
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, K2, subAlphas, subAlphaSum);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, K2, subAlphas, subAlphaSum);

		PAModel(const PAArgs& args)
			: BaseClass(args), K2(args.k2)
		{
			if (K2 == 0 || K2 >= 0x80000000) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong K2 value (K2 = %zd)", K2));

			if (args.subalpha.size() == 1)
			{
				subAlphas = Matrix::Constant(args.k, args.k2, args.subalpha[0]);
			}
			else if(args.subalpha.size() == args.k2)
			{
				subAlphas = Eigen::Map<const Eigen::Matrix<Float, 1, -1>>(args.subalpha.data(), args.subalpha.size()).replicate(args.k, 1);
			}
			else
			{
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong subalpha value (len = %zd)", args.subalpha.size()));
			}
			subAlphaSum = subAlphas.rowwise().sum();
			this->optimInterval = 1;
		}

		GETTER(K2, size_t, K2);
		GETTER(DirichletEstIteration, size_t, iteration);

		void setDirichletEstIteration(size_t iter) override
		{
			if (!iter) throw exc::InvalidArgument("iter must > 0");
			iteration = iter;
		}

		Float getSubAlpha(Tid k1, Tid k2) const override { return subAlphas(k1, k2); }

		std::vector<Float> getSubAlpha(Tid k1) const override
		{
			std::vector<Float> ret(K2);
			Eigen::Map<Eigen::VectorXf>{ret.data(), (Eigen::Index)ret.size()} = subAlphas.row(k1).transpose();
			return ret;
		}

		std::vector<Float> getSubTopicBySuperTopic(Tid k, bool normalize) const override
		{
			assert(k < this->K);
			std::vector<Float> ret(K2);
			Float sum = this->globalState.numByTopic[k] + subAlphaSum[k];
			if (!normalize) sum = 1;
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), K2 };
			m = (this->globalState.numByTopic1_2.row(k).array().template cast<Float>() + subAlphas.row(k).array()) / sum;
			return ret;
		}

		std::vector<std::pair<Tid, Float>> getSubTopicBySuperTopicSorted(Tid k, size_t topN) const override
		{
			return extractTopN<Tid>(getSubTopicBySuperTopic(k, true), topN);
		}

		std::vector<Float> getSubTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			std::vector<Float> ret(K2);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), K2 };
			if (normalize)
			{
				m = ((doc.numByTopic1_2.array().template cast<Float>() + subAlphas.array()).colwise().sum()) / (doc.getSumWordWeight() + subAlphas.sum());
			}
			else
			{
				m = (doc.numByTopic1_2.array().template cast<Float>() + subAlphas.array()).colwise().sum();
			}
			return ret;
		}

		std::vector<Float> getSubTopicsByDoc(const DocumentBase* doc, bool normalize) const override
		{
			return static_cast<const DerivedClass*>(this)->getSubTopicsByDoc(*static_cast<const _DocType*>(doc), normalize);
		}

		std::vector<std::pair<Tid, Float>> getSubTopicsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return extractTopN<Tid>(getSubTopicsByDoc(doc, true), topN);
		}

		std::vector<Float> _getWidsByTopic(Tid k2, bool normalize = true) const
		{
			assert(k2 < K2);
			const size_t V = this->realV;
			std::vector<Float> ret(V);
			Float sum = this->globalState.numByTopic2[k2] + V * this->eta;
			if (!normalize) sum = 1;
			auto r = this->globalState.numByTopicWord.row(k2);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + this->eta) / sum;
			}
			return ret;
		}

		size_t getNumTopicsForPrior() const override
		{
			return this->K2;
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			if (priors.size() != K2) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors.size() must be equal to K2.");
			for (auto p : priors)
			{
				if (p < 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors must not be less than 0.");
			}
			this->dict.add(word);
			this->etaByWord.emplace(word, priors);
		}

		std::vector<uint64_t> getCountBySuperTopic() const override
		{
			std::vector<uint64_t> cnt(this->K);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Zs[i]];
				}
			}
			return cnt;
		}
	};
	
	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentPA<_tw>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		DocumentLDA<_tw>::update(ptr, mdl);
		numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(mdl.getK(), mdl.getK2());
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numByTopic1_2(this->Zs[i], Z2s[i]) += _tw != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}