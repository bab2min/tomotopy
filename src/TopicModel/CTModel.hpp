#pragma once
#include "LDAModel.hpp"
#include "../Utils/MultiNormalDistribution.hpp"
#include "../Utils/TruncMultiNormal.hpp"
#include "CT.h"
/*
Implementation of CTM using Gibbs sampling by bab2min
* Blei, D., & Lafferty, J. (2006). Correlated topic models. Advances in neural information processing systems, 18, 147.
* Mimno, D., Wallach, H., & McCallum, A. (2008, December). Gibbs sampling for logistic normal topic models with graph-based priors. In NIPS Workshop on Analyzing Graphs (Vol. 61).
*/

namespace tomoto
{
	template<TermWeight _TW>
	struct ModelStateCTM : public ModelStateLDA<_TW>
	{
	};

	template<TermWeight _TW, size_t _Flags = 0,
		typename _Interface = ICTModel,
		typename _Derived = void,
		typename _DocType = DocumentCTM<_TW>,
		typename _ModelState = ModelStateCTM<_TW>>
	class CTModel : public LDAModel<_TW, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, CTModel<_TW, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "CTM";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, CTModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		size_t numBetaSample = 10;
		size_t numTMNSample = 5;
		size_t numDocBetaSample = -1;
		math::MultiNormalDistribution<FLOAT> topicPrior;
		
		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = doc.smBeta.array()
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);
			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void updateBeta(_DocType& doc, RandGen& rg) const
		{
			Eigen::Matrix<FLOAT, -1, 1> pbeta, lowerBound, upperBound;
			constexpr FLOAT epsilon = 1e-8;
			constexpr size_t burnIn = 3;
			pbeta = lowerBound = upperBound = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K);
			for (size_t i = 0; i < numBetaSample + burnIn; ++i)
			{
				if (i == 0) pbeta = Eigen::Matrix<FLOAT, -1, 1>::Ones(this->K);
				else pbeta = doc.beta.col(i % numBetaSample).array().exp();
				FLOAT betaESum = pbeta.sum() + 1;
				pbeta /= betaESum;
				for (size_t k = 0; k < this->K; ++k)
				{
					FLOAT N_k = doc.numByTopic[k] + this->alpha;
					FLOAT N_nk = doc.getSumWordWeight() + this->alpha * (this->K + 1) - N_k;
					FLOAT u1 = std::generate_canonical<FLOAT, 32>(rg), u2 = std::generate_canonical<FLOAT, 32>(rg);
					FLOAT max_uk = epsilon + pow(u1, (FLOAT)1 / N_k)  * (pbeta[k] - epsilon);
					FLOAT min_unk = (1 - pow(u2, (FLOAT)1 / N_nk))
						* (1 - pbeta[k]) + pbeta[k];

					FLOAT c = betaESum * (1 - pbeta[k]);
					lowerBound[k] = log(c * max_uk / (1 - max_uk));
					upperBound[k] = log(c * min_unk / (1 - min_unk));
					if (lowerBound[k] > upperBound[k])
					{
						THROW_ERROR_WITH_INFO(exception::TrainingError,
							text::format("Bound Error: LB(%f) > UB(%f)\n"
								"max_uk: %f, min_unk: %f, c: %f", lowerBound[k], upperBound[k], max_uk, min_unk, c));
					}
				}
				try
				{
					math::sampleFromTruncatedMultiNormal(doc.beta.col((i + 1) % numBetaSample),
						topicPrior, lowerBound, upperBound, rg, numTMNSample);

					if (!std::isfinite(doc.beta.col((i + 1) % numBetaSample)[0])) 
						THROW_ERROR_WITH_INFO(exception::TrainingError,
							text::format("doc.beta.col(%d) is %f", (i + 1) % numBetaSample, 
							doc.beta.col((i + 1) % numBetaSample)[0]));
				}
				catch (const std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
					THROW_ERROR_WITH_INFO(exception::TrainingError, e.what());
				}
			}

			// update softmax-applied beta coefficient
			doc.smBeta.head(this->K) = doc.beta.block(0, 0, this->K, std::min(numBetaSample, numDocBetaSample)).rowwise().mean();
			doc.smBeta = doc.smBeta.array().exp();
			doc.smBeta /= doc.smBeta.array().sum();
		}

		void sampleDocument(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
			BaseClass::sampleDocument(doc, docId, ld, rgs, iterationCnt);
			if (iterationCnt >= this->burnIn && this->optimInterval && (iterationCnt + 1) % this->optimInterval == 0)
			{
				updateBeta(doc, rgs);
			}
		}

		int restoreFromTrainingError(const exception::TrainingError& e, ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			std::cerr << "Failed to sample! Reset prior and retry!" << std::endl;
			const size_t chStride = std::min(pool.getNumWorkers() * 8, this->docs.size());
			topicPrior = math::MultiNormalDistribution<FLOAT>{ this->K };
			std::vector<std::future<void>> res;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this](size_t threadId, size_t ch)
				{
					for (size_t i = ch; i < this->docs.size(); i += chStride)
					{
						this->docs[i].beta.setZero();
						updateBeta(this->docs[i], rgs[threadId]);
					}
				}, ch));
			}
			for (auto&& r : res) r.get();
			return 0;
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			std::vector<std::future<void>> res;
			topicPrior = math::MultiNormalDistribution<FLOAT>::estimate([this](size_t i)
			{
				return this->docs[i / numBetaSample].beta.col(i % numBetaSample);
			}, this->docs.size() * numBetaSample);
			if (!std::isfinite(topicPrior.mean[0]))
				THROW_ERROR_WITH_INFO(exception::TrainingError, 
					text::format("topicPrior.mean is %f", topicPrior.mean[0]));
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			const auto alpha = this->alpha;

			double ll = 0;
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				Eigen::Matrix<FLOAT, -1, 1> pbeta = doc.smBeta.array().log();
				FLOAT last = pbeta[K - 1];
				for (TID k = 0; k < K; ++k)
				{
					ll += pbeta[k] * (doc.numByTopic[k] + alpha) - math::lgammaT(doc.numByTopic[k] + alpha + 1);
				}
				pbeta.array() -= last;
				ll += topicPrior.getLL(pbeta.head(this->K));
				ll += math::lgammaT(doc.getSumWordWeight() + alpha * K + 1);
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
			doc.beta = Eigen::Matrix<FLOAT, -1, -1>::Zero(this->K, numBetaSample);
			doc.smBeta = Eigen::Matrix<FLOAT, -1, 1>::Constant(this->K, (FLOAT)1 / this->K);
		}

		void updateDocs()
		{
			BaseClass::updateDocs();
			for (auto& doc : this->docs)
			{
				doc.beta = Eigen::Matrix<FLOAT, -1, -1>::Zero(this->K, numBetaSample);
			}
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			if (initDocs)
			{
				topicPrior = math::MultiNormalDistribution<FLOAT>{ this->K };
			}
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, numBetaSample, numTMNSample, topicPrior);

	public:
		CTModel(size_t _K = 1, FLOAT smoothingAlpha = 0.1, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, smoothingAlpha, _eta, _rg)
		{
			this->optimInterval = 2;
		}

		std::vector<FLOAT> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<FLOAT> ret(this->K);
			Eigen::Map<Eigen::Matrix<FLOAT, -1, 1>>{ret.data(), this->K}.array() =
				doc.numByTopic.array().template cast<FLOAT>() / doc.getSumWordWeight();
			return ret;
		}

		std::vector<FLOAT> getPriorMean() const
		{
			return { topicPrior.mean.data(), topicPrior.mean.data() + topicPrior.mean.size() };
		}

		std::vector<FLOAT> getPriorCov() const
		{
			return { topicPrior.cov.data(), topicPrior.cov.data() + topicPrior.cov.size() };
		}

		std::vector<FLOAT> getCorrelationTopic(TID k) const
		{
			Eigen::Matrix<FLOAT, -1, 1> ret = topicPrior.cov.col(k).array() / (topicPrior.cov.diagonal().array() * topicPrior.cov(k, k)).sqrt();
			return { ret.data(), ret.data() + ret.size() };
		}

		GETTER(NumBetaSample, size_t, numBetaSample);

		void setNumBetaSample(size_t _numSample)
		{
			numBetaSample = _numSample;
		}

		GETTER(NumDocBetaSample, size_t, numDocBetaSample);

		void setNumDocBetaSample(size_t _numSample)
		{
			numDocBetaSample = _numSample;
		}

		GETTER(NumTMNSample, size_t, numTMNSample);

		void setNumTMNSample(size_t _numSample)
		{
			numTMNSample = _numSample;
		}
	};
}
