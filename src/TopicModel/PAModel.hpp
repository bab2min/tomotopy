#pragma once
#include "LDAModel.hpp"
#include "PA.h"

/*
Implementation of Pachinko Allocation using Gibbs sampling by bab2min

Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.
*/

namespace tomoto
{
	template<TermWeight _TW>
	struct ModelStatePA : public ModelStateLDA<_TW>
	{
		using WeightType = typename ModelStateLDA<_TW>::WeightType;
		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;
		Eigen::Matrix<WeightType, -1, 1> numByTopic2;
		Eigen::Matrix<FLOAT, -1, 1> subTmp;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_TW>, numByTopic1_2, numByTopic2);
	};

	template<TermWeight _TW,
		typename _Interface = IPAModel,
		typename _Derived = void,
		typename _DocType = DocumentPA<_TW>,
		typename _ModelState = ModelStatePA<_TW>>
	class PAModel : public LDAModel<_TW, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		size_t K2;
		FLOAT epsilon = 1e-5;
		size_t iteration = 5;

		Eigen::Matrix<FLOAT, -1, 1> subAlphaSum; // len = K
		Eigen::Matrix<FLOAT, -1, -1> subAlphas; // len = K * K2
		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			const auto K = this->K;
			std::vector<std::future<void>> res;
			for (size_t k = 0; k < K; ++k)
			{
				pool.enqueue([&, k](size_t)
				{
					for (size_t i = 0; i < iteration; ++i)
					{
						FLOAT denom = this->template calcDigammaSum<>([&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), subAlphaSum[k]);
						for (size_t k2 = 0; k2 < K2; ++k2)
						{
							FLOAT nom = this->template calcDigammaSum<>([&](size_t i) { return this->docs[i].numByTopic1_2(k, k2); }, this->docs.size(), subAlphas(k, k2));
							subAlphas(k, k2) = std::max(nom / denom * subAlphas(k, k2), epsilon);
						}
						subAlphaSum[k] = subAlphas.row(k).sum();
					}
				});
			}
			for (auto& r : res) r.get();
		}

		// topic 1 & 2 assignment likelihoods for new word. ret K*K2 FLOATs
		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			const auto eta = this->eta;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			
			ld.subTmp = (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + eta) / (ld.numByTopic2.array().template cast<FLOAT>() + V * eta);

			for (size_t k = 0; k < this->K; ++k)
			{
				zLikelihood.segment(K2 * k, K2) = (doc.numByTopic[k] + this->alpha)
					* (doc.numByTopic1_2.row(k).transpose().array().template cast<FLOAT>() + subAlphas.row(k).transpose().array()) / (doc.numByTopic[k] + subAlphaSum[k])
					* ld.subTmp.array();
			}
			sample::prefixSum(zLikelihood.data(), zLikelihood.size());
			return &zLikelihood[0];
		}

		template<int INC> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID z1, TID z2) const
		{
			assert(vid < this->realV);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[z1], INC * weight);
			updateCnt<DEC>(doc.numByTopic1_2(z1, z2), INC * weight);
			updateCnt<DEC>(ld.numByTopic[z1], INC * weight);
			updateCnt<DEC>(ld.numByTopic2[z2], INC * weight);
			updateCnt<DEC>(ld.numByTopic1_2(z1, z2), INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(z2, vid), INC * weight);
		}

		void sampleDocument(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.Z2s[w]);
				auto dist = getZLikelihoods(ld, doc, docId, doc.words[w]);
				auto z = sample::sampleFromDiscreteAcc(dist, dist + this->K * K2, rgs);
				doc.Zs[w] = z / K2;
				doc.Z2s[w] = z % K2;
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.Z2s[w]);
			}
		}

		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, RandGen*) const
		{
			std::vector<std::future<void>> res(pool.getNumWorkers());

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
			if (_TW != TermWeight::one)
			{
				globalState.numByTopic = globalState.numByTopic.cwiseMax(0);
				globalState.numByTopic1_2 = globalState.numByTopic1_2.cwiseMax(0);
				globalState.numByTopic2 = globalState.numByTopic2.cwiseMax(0);
				globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
			}

			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, this, i](size_t threadId)
				{
					localData[i] = globalState;
				});
			}
			for (auto&& r : res) r.get();
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
				for (TID k = 0; k < K; ++k)
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
			for (TID k = 0; k < K; ++k)
			{
				ll += math::lgammaT(subAlphaSum[k]);
				ll -= math::lgammaT(ld.numByTopic[k] + subAlphaSum[k]);
				for (TID k2 = 0; k2 < K2; ++k2)
				{
					ll -= math::lgammaT(subAlphas(k, k2));
					ll += math::lgammaT(ld.numByTopic1_2(k, k2) + subAlphas(k, k2));
				}
			}
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K2;
			for (TID k2 = 0; k2 < K2; ++k2)
			{
				ll -= math::lgammaT(ld.numByTopic2[k2] + V * eta);
				for (VID v = 0; v < V; ++v)
				{
					ll += math::lgammaT(ld.numByTopicWord(k2, v) + eta);
				}
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);

			doc.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
			doc.Z2s = tvector<TID>(wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K * K2);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
				this->globalState.numByTopic2 = Eigen::Matrix<WeightType, -1, 1>::Zero(K2);
				this->globalState.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K2, V);
			}
		}

		struct Generator
		{
			std::uniform_int_distribution<TID> theta, theta2;
		};

		Generator makeGeneratorForInit() const
		{
			return Generator{ 
				std::uniform_int_distribution<TID>{0, (TID)(this->K - 1)},
				std::uniform_int_distribution<TID>{0, (TID)(K2 - 1)},
			};
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto w = doc.words[i];
			doc.Zs[i] = g.theta(rgs);
			doc.Z2s[i] = g.theta2(rgs);
			addWordTo<1>(ld, doc, i, w, doc.Zs[i], doc.Z2s[i]);
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, K2, subAlphas, subAlphaSum);

	public:
		PAModel(size_t _K1 = 1, size_t _K2 = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K1, _alpha, _eta, _rg), K2(_K2)
		{
			if (_K2 == 0 || _K2 >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong K2 value (K2 = %zd)", _K2));
			subAlphaSum = Eigen::Matrix<FLOAT, -1, 1>::Constant(_K1, _K2 * 0.1);
			subAlphas = Eigen::Matrix<FLOAT, -1, -1>::Constant(_K1, _K2, 0.1);
			this->optimInterval = 1;
		}

		GETTER(K2, size_t, K2);
		GETTER(DirichletEstIteration, size_t, iteration);

		void setDirichletEstIteration(size_t iter) override
		{
			if (!iter) throw std::invalid_argument("iter must > 0");
			iteration = iter;
		}

		FLOAT getSubAlpha(TID k1, TID k2) const override { return subAlphas(k1, k2); }

		std::vector<FLOAT> getSubTopicBySuperTopic(TID k) const
		{
			assert(k < this->K);
			FLOAT sum = this->globalState.numByTopic[k] + subAlphaSum[k];
			Eigen::Matrix<FLOAT, -1, 1> ret = (this->globalState.numByTopic1_2.row(k).array().template cast<FLOAT>() + subAlphas.row(k).array()) / sum;
			return { ret.data(), ret.data() + K2 };
		}

		std::vector<std::pair<TID, FLOAT>> getSubTopicBySuperTopicSorted(TID k, size_t topN) const
		{
			return extractTopN<TID>(getSubTopicBySuperTopic(k), topN);
		}

		std::vector<FLOAT> _getWidsByTopic(TID k2) const
		{
			assert(k2 < K2);
			const size_t V = this->realV;
			std::vector<FLOAT> ret(V);
			FLOAT sum = this->globalState.numByTopic2[k2] + V * this->eta;
			auto r = this->globalState.numByTopicWord.row(k2);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + this->eta) / sum;
			}
			return ret;
		}
	};
	
	template<TermWeight _TW>
	template<typename _TopicModel>
	void DocumentPA<_TW>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		DocumentLDA<_TW>::update(ptr, mdl);
		numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(mdl.getK(), mdl.getK2());
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numByTopic1_2(this->Zs[i], Z2s[i]) += _TW != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}