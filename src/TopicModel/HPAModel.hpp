#pragma once
#include "PAModel.hpp"
#include "HPA.h"
/*
Implementation of Hierarchical Pachinko Allocation using Gibbs sampling by bab2min

Mimno, D., Li, W., & McCallum, A. (2007, June). Mixtures of hierarchical topics with pachinko allocation. In Proceedings of the 24th international conference on Machine learning (pp. 633-640). ACM.
*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateHPA : public ModelStateLDA<_tw>
	{
		using WeightType = typename ModelStateLDA<_tw>::WeightType;

		std::array<Eigen::Matrix<WeightType, -1, -1>, 3> numByTopicWord;
		std::array<Eigen::Matrix<WeightType, -1, 1>, 3> numByTopic;
		std::array<Vector, 2> subTmp;

		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_tw>, numByTopicWord, numByTopic, numByTopic1_2);
	};

	template<TermWeight _tw, typename _RandGen,
		bool _Exclusive = false,
		typename _Interface = IHPAModel,
		typename _Derived = void,
		typename _DocType = DocumentHPA<_tw>,
		typename _ModelState = ModelStateHPA<_tw>
	>
	class HPAModel : public LDAModel<_tw, _RandGen, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, HPAModel<_tw, _RandGen, _Exclusive>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, HPAModel<_tw, _RandGen, _Exclusive>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		uint64_t K2;
		Float epsilon = 0.00001;
		size_t iteration = 5;

		//Vector alphas; // len = (K + 1)

		Vector subAlphaSum; // len = K
		Matrix subAlphas; // len = K * (K2 + 1)

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			const auto K = this->K;
			for (size_t i = 0; i < iteration; ++i)
			{
				Float denom = this->template calcDigammaSum<>(&pool, [&](size_t i) { return this->docs[i].getSumWordWeight(); }, this->docs.size(), this->alphas.sum());

				for (size_t k = 0; k <= K; ++k)
				{
					Float nom = this->template calcDigammaSum<>(&pool, [&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), this->alphas[k]);
					this->alphas[k] = std::max(nom / denom * this->alphas[k], epsilon);
				}
			}

			std::vector<std::future<void>> res;
			for (size_t k = 0; k < K; ++k)
			{
				res.emplace_back(pool.enqueue([&, k](size_t)
				{
					for (size_t i = 0; i < iteration; ++i)
					{
						Float denom = this->template calcDigammaSum<>(nullptr, [&](size_t i) { return this->docs[i].numByTopic[k + 1]; }, this->docs.size(), subAlphaSum[k]);
						for (size_t k2 = 0; k2 <= K2; ++k2)
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

		std::pair<size_t, size_t> getRangeOfK(size_t k) const
		{
			return std::make_pair<size_t, size_t>(ceil(k * (float)K2 / this->K), ceil((k + 1) * (float)K2 / this->K));
		}

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			const auto K = this->K;
			const auto eta = this->eta;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;

			Float rootWordProb = (ld.numByTopicWord[0](0, vid) + eta) / (ld.numByTopic[0](0) + V * eta);
			ld.subTmp[0] = (ld.numByTopicWord[1].col(vid).array().template cast<Float>() + eta) / (ld.numByTopic[1].array().template cast<Float>() + V * eta);
			ld.subTmp[1] = (ld.numByTopicWord[2].col(vid).array().template cast<Float>() + eta) / (ld.numByTopic[2].array().template cast<Float>() + V * eta);

			if (_Exclusive)
			{
				for (size_t k = 0; k < K; ++k)
				{
					auto r = getRangeOfK(k);
					auto r1 = r.first, r2 = r.second;
					zLikelihood.segment(r1, r2 - r1) = (doc.numByTopic[k + 1] + this->alphas[k + 1])
						* (doc.numByTopic1_2.row(k).segment(r1 + 1, r2 - r1).array().transpose().template cast<Float>() + subAlphas.row(k).segment(r1 + 1, r2 - r1).array().transpose())
						/ (doc.numByTopic[k + 1] + subAlphaSum[k])
						* ld.subTmp[1].segment(r1, r2 - r1).array();
				}

				zLikelihood.segment(K2, K) = (doc.numByTopic.tail(K).array().template cast<Float>() + this->alphas.tail(K).array())
					* (doc.numByTopic1_2.col(0).array().template cast<Float>() + subAlphas.col(0).array())
					/ (doc.numByTopic.tail(K).array().template cast<Float>() + subAlphaSum.array().template cast<Float>())
					* ld.subTmp[0].array();

				zLikelihood[K2 + K] = (doc.numByTopic[0] + this->alphas[0]) * rootWordProb;
			}
			else
			{
				for (size_t k = 0; k < K; ++k)
				{
					zLikelihood.segment(K2 * k, K2) = (doc.numByTopic[k + 1] + this->alphas[k + 1])
						* (doc.numByTopic1_2.row(k).tail(K2).array().transpose().template cast<Float>() + subAlphas.row(k).tail(K2).array().transpose()) 
						/ (doc.numByTopic[k + 1] + subAlphaSum[k])
						* ld.subTmp[1].array();
				}

				zLikelihood.segment(K2 * K, K) = (doc.numByTopic.tail(K).array().template cast<Float>() + this->alphas.tail(K).array())
					* (doc.numByTopic1_2.col(0).array().template cast<Float>() + subAlphas.col(0).array())
					/ (doc.numByTopic.tail(K).array().template cast<Float>() + subAlphaSum.array().template cast<Float>())
					* ld.subTmp[0].array();

				zLikelihood[K2 * K + K] = (doc.numByTopic[0] + this->alphas[0]) * rootWordProb;
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
			if (z1)
			{
				updateCnt<_dec>(doc.numByTopic1_2(z1 - 1, z2), _inc * weight);
				updateCnt<_dec>(ld.numByTopic1_2(z1 - 1, z2), _inc * weight);
			}

			if (!z1)
			{
				updateCnt<_dec>(ld.numByTopic[0][0], _inc * weight);
				updateCnt<_dec>(ld.numByTopicWord[0](0, vid), _inc * weight);
				
			}
			else if (!z2)
			{
				updateCnt<_dec>(ld.numByTopic[1][z1 - 1], _inc * weight);
				updateCnt<_dec>(ld.numByTopicWord[1](z1 - 1, vid), _inc * weight);
			}
			else
			{
				updateCnt<_dec>(ld.numByTopic[2][z2 - 1], _inc * weight);
				updateCnt<_dec>(ld.numByTopicWord[2](z2 - 1, vid), _inc * weight);
			}
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

			const auto K = this->K;
			for (size_t w = b; w < e; ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w], doc.Z2s[w]);
				Float* dist;
				if (this->etaByTopicWord.size())
				{
					THROW_ERROR_WITH_INFO(exc::Unimplemented, "Unimplemented features");
				}
				else
				{
					dist = getZLikelihoods<false>(ld, doc, docId, doc.words[w] - vOffset);
				}
				if (_Exclusive)
				{
					auto z = sample::sampleFromDiscreteAcc(dist, dist + K2 + K + 1, rgs);
					if (z < K2)
					{
						doc.Zs[w] = (z * K / K2) + 1;
						doc.Z2s[w] = z + 1;
					}
					else if (z < K2 + K)
					{
						doc.Zs[w] = z - K2 + 1;
						doc.Z2s[w] = 0;
					}
					else
					{
						doc.Zs[w] = 0;
						doc.Z2s[w] = 0;
					}
				}
				else
				{
					auto z = sample::sampleFromDiscreteAcc(dist, dist + K * K2 + K + 1, rgs);
					if (z < K * K2)
					{
						doc.Zs[w] = (z / K2) + 1;
						doc.Z2s[w] = (z % K2) + 1;
					}
					else if (z < K * K2 + K)
					{
						doc.Zs[w] = z - K * K2 + 1;
						doc.Z2s[w] = 0;
					}
					else
					{
						doc.Zs[w] = 0;
						doc.Z2s[w] = 0;
					}
				}
				addWordTo<1>(ld, doc, w, doc.words[w] - vOffset, doc.Zs[w], doc.Z2s[w]);
			}
		}

		template<typename _ExtraDocData>
		void distributePartition(ThreadPool& pool, const _ModelState& globalState, _ModelState* localData, const _ExtraDocData& edd) const
		{
		}

		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, _RandGen*, const _ExtraDocData& edd) const
		{
			tState = globalState;
			globalState = localData[0];
			for (size_t i = 1; i < pool.getNumWorkers(); ++i)
			{
				globalState.numByTopic[0] += localData[i].numByTopic[0] - tState.numByTopic[0];
				globalState.numByTopic[1] += localData[i].numByTopic[1] - tState.numByTopic[1];
				globalState.numByTopic[2] += localData[i].numByTopic[2] - tState.numByTopic[2];
				globalState.numByTopic1_2 += localData[i].numByTopic1_2 - tState.numByTopic1_2;
				globalState.numByTopicWord[0] += localData[i].numByTopicWord[0] - tState.numByTopicWord[0];
				globalState.numByTopicWord[1] += localData[i].numByTopicWord[1] - tState.numByTopicWord[1];
				globalState.numByTopicWord[2] += localData[i].numByTopicWord[2] - tState.numByTopicWord[2];
			}

			// make all count being positive
			if (_tw != TermWeight::one)
			{
				globalState.numByTopic[0] = globalState.numByTopic[0].cwiseMax(0);
				globalState.numByTopic[1] = globalState.numByTopic[1].cwiseMax(0);
				globalState.numByTopic[2] = globalState.numByTopic[2].cwiseMax(0);
				globalState.numByTopic1_2 = globalState.numByTopic1_2.cwiseMax(0);
				globalState.numByTopicWord[0] = globalState.numByTopicWord[0].cwiseMax(0);
				globalState.numByTopicWord[1] = globalState.numByTopicWord[1].cwiseMax(0);
				globalState.numByTopicWord[2] = globalState.numByTopicWord[2].cwiseMax(0);
			}
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(1 + this->K + K2);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] >= this->realV) continue;
					if (doc.Zs[i] == 0 && doc.Z2s[i] == 0)
					{
						++cnt[0];
					}
					else if (doc.Zs[i] && doc.Z2s[i] == 0)
					{
						++cnt[doc.Zs[i]];
					}
					else
					{
						++cnt[this->K + doc.Z2s[i]];
					}
				}
			}
			return cnt;
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			const auto alphaSum = this->alphas.sum();

			double ll = 0;
			ll = math::lgammaT(alphaSum);
			for (size_t k = 0; k < K; ++k) ll -= math::lgammaT(this->alphas[k]);
			ll *= std::distance(_first, _last);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll -= math::lgammaT(doc.getSumWordWeight() + alphaSum);
				for (Tid k = 0; k <= K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + this->alphas[k]);
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
				ll -= math::lgammaT(ld.numByTopic1_2.row(k).sum() + subAlphaSum[k]);
				for (Tid k2 = 0; k2 <= K2; ++k2)
				{
					ll -= math::lgammaT(subAlphas(k, k2));
					ll += math::lgammaT(ld.numByTopic1_2(k, k2) + subAlphas(k, k2));
				}
			}
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * (K2 + K + 1);

			ll -= math::lgammaT(ld.numByTopic[0][0] + V * eta);
			for (Vid v = 0; v < V; ++v)
			{
				ll += math::lgammaT(ld.numByTopicWord[0](0, v) + eta);
			}
			for (Tid k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[1][k] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
					ll += math::lgammaT(ld.numByTopicWord[1](k, v) + eta);
				}
			}
			for (Tid k2 = 0; k2 < K2; ++k2)
			{
				ll -= math::lgammaT(ld.numByTopic[2][k2] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
					ll += math::lgammaT(ld.numByTopicWord[2](k2, v) + eta);
				}
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			doc.numByTopic.init(nullptr, this->K + 1, 1);
			doc.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2 + 1);
			doc.Zs = tvector<Tid>(wordSize);
			doc.Z2s = tvector<Tid>(wordSize);
			if (_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Vector::Zero(1 + this->K + this->K * K2);
			if (initDocs)
			{
				this->globalState.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2 + 1);
				this->globalState.numByTopic[0] = Eigen::Matrix<WeightType, -1, 1>::Zero(1);
				this->globalState.numByTopic[1] = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
				this->globalState.numByTopic[2] = Eigen::Matrix<WeightType, -1, 1>::Zero(K2);
				this->globalState.numByTopicWord[0] = Eigen::Matrix<WeightType, -1, -1>::Zero(1, V);
				this->globalState.numByTopicWord[1] = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, V);
				this->globalState.numByTopicWord[2] = Eigen::Matrix<WeightType, -1, -1>::Zero(K2, V);
			}
		}

		struct Generator
		{
			std::uniform_int_distribution<Tid> theta, theta2;
			std::discrete_distribution<> level;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			return Generator{
				std::uniform_int_distribution<Tid>{1, (Tid)(this->K)},
				std::uniform_int_distribution<Tid>{1, (Tid)(K2)},
				std::discrete_distribution<>{1.0, 1.0, 1.0},
			};
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto w = doc.words[i];
			switch (g.level(rgs))
			{
			case 0:
				doc.Zs[i] = 0;
				doc.Z2s[i] = 0;
				break;
			case 1:
				doc.Zs[i] = g.theta(rgs);
				doc.Z2s[i] = 0;
				break;
			default:
				if (_Exclusive)
				{
					doc.Z2s[i] = g.theta2(rgs);
					doc.Zs[i] = (doc.Z2s[i] - 1) * this->K / K2 + 1;
				}
				else
				{
					doc.Zs[i] = g.theta(rgs);
					doc.Z2s[i] = g.theta2(rgs);
				}
			}
			addWordTo<1>(ld, doc, i, w, doc.Zs[i], doc.Z2s[i]);
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, K2, subAlphas, subAlphaSum);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, K2, subAlphas, subAlphaSum);

		HPAModel(const HPAArgs& args)
			: BaseClass(args, false), K2(args.k2)
		{
			if (K2 == 0 || K2 >= 0x80000000) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong K2 value (K2 = %zd)", K2));

			if (args.alpha.size() == 1)
			{
				this->alphas = Vector::Constant(args.k + 1, args.alpha[0]);
			}
			else if (args.alpha.size() == args.k + 1)
			{
				this->alphas = Eigen::Map<const Vector>(args.alpha.data(), (Eigen::Index)args.alpha.size());
			}
			else
			{
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong alpha value (len = %zd)", args.alpha.size()));
			}

			if (args.subalpha.size() == 1)
			{
				subAlphas = Matrix::Constant(args.k, args.k2 + 1, args.subalpha[0]);
			}
			else if (args.subalpha.size() == args.k2 + 1)
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

		Float getSubAlpha(Tid k1, Tid k2) const override 
		{ 
			if (_Exclusive)
			{
				if (k2 && k1 != (k2 - 1) * this->K / K2) return 0;
			}
			return subAlphas(k1, k2); 
		}

		std::vector<Float> getSubAlpha(Tid k1) const override
		{
			std::vector<Float> ret(K2 + 1);
			Eigen::Map<Eigen::VectorXf>{ret.data(), (Eigen::Index)ret.size()} = subAlphas.row(k1).transpose();
			return ret;
		}

		std::vector<Float> getSubTopicBySuperTopic(Tid k, bool normalize) const override
		{
			std::vector<Float> ret(K2);
			assert(k < this->K);
			Float sum = this->globalState.numByTopic1_2.row(k).sum() + subAlphaSum[k];
			if (!normalize) sum = 1;
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), (Eigen::Index)K2 };
			m = (this->globalState.numByTopic1_2.row(k).segment(1, K2).array().template cast<Float>() + subAlphas.row(k).segment(1, K2).array()) / sum;
			return ret;
		}

		std::vector<std::pair<Tid, Float>> getSubTopicBySuperTopicSorted(Tid k, size_t topN) const override
		{
			return extractTopN<Tid>(getSubTopicBySuperTopic(k, true), topN);
		}

		std::vector<Float> _getWidsByTopic(Tid k, bool normalize = true) const
		{
			const size_t V = this->realV;
			std::vector<Float> ret(V);
			size_t level = 0;
			if (k >= 1)
			{
				++level;
				k -= 1;
				if (k >= this->K)
				{
					++level;
					k -= this->K;
				}
			}
			Float sum = this->globalState.numByTopic[level][k] + V * this->eta;
			if (!normalize) sum = 1;
			auto r = this->globalState.numByTopicWord[level].row(k);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + this->eta) / sum;
			}
			return ret;
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(1 + this->K + K2);
			Float sum = doc.getSumWordWeight() + this->alphas.sum();
			if (!normalize) sum = 1;

			ret[0] = (doc.numByTopic[0] + this->alphas[0]) / sum;
			for (size_t k = 0; k < this->K; ++k)
			{
				ret[k + 1] = (doc.numByTopic1_2(k, 0) + subAlphas(k, 0)) / sum;
			}
			for (size_t k = 0; k < K2; ++k)
			{
				ret[k + this->K + 1] = doc.numByTopic1_2.col(k + 1).sum() / sum;
			}
			return ret;
		}

		std::vector<Float> getSubTopicsByDoc(const DocumentBase* doc, bool normalize) const override
		{
			throw std::runtime_error{ "not applicable" };
		}

		std::vector<std::pair<Tid, Float>> getSubTopicsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			throw std::runtime_error{ "not applicable" };
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			THROW_ERROR_WITH_INFO(exc::Unimplemented, "HPAModel doesn't provide setWordPrior function.");
		}

		std::vector<uint64_t> getCountBySuperTopic() const override
		{
			std::vector<uint64_t> cnt(this->K);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] >= this->realV) continue;
					if (doc.Zs[i] && doc.Z2s[i] == 0)
					{
						++cnt[doc.Zs[i] - 1];
					}
				}
			}
			return cnt;
		}
	};

	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentHPA<_tw>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		this->numByTopic.init(ptr, mdl.getK() + 1, 1);
		this->numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(mdl.getK(), mdl.getK2() + 1);
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			this->numByTopic[this->Zs[i]] += _tw != TermWeight::one ? this->wordWeights[i] : 1;
			if (this->Zs[i]) this->numByTopic1_2(this->Zs[i] - 1, this->Z2s[i]) += _tw != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}

}
