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
	template<TermWeight _tw>
	struct ModelStateCTM : public ModelStateLDA<_tw>
	{
	};

	template<TermWeight _tw, typename _RandGen,
		size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = ICTModel,
		typename _Derived = void,
		typename _DocType = DocumentCTM<_tw>,
		typename _ModelState = ModelStateCTM<_tw>>
	class CTModel : public LDAModel<_tw, _RandGen, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, CTModel<_tw, _RandGen, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, CTModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr auto tmid()
		{
			return serializer::to_key("CTM\0");
		}

		uint64_t numBetaSample = 10;
		uint64_t numTMNSample = 5;
		uint64_t numDocBetaSample = -1;
		math::MultiNormalDistribution<Float> topicPrior;
		
		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = doc.smBeta.array()
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());
			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void updateBeta(_DocType& doc, _RandGen& rg) const
		{
			Vector pbeta, lowerBound, upperBound;
			constexpr Float epsilon = 1e-8;
			constexpr size_t burnIn = 3;

			pbeta = lowerBound = upperBound = Vector::Zero(this->K);
			for (size_t i = 0; i < numBetaSample + burnIn; ++i)
			{
				if (i == 0) pbeta = Vector::Ones(this->K);
				else pbeta = doc.beta.col(i % numBetaSample).array().exp();

				Float betaESum = pbeta.sum() + 1;
				pbeta /= betaESum;
				for (size_t k = 0; k < this->K; ++k)
				{
					Float N_k = doc.numByTopic[k] + this->alphas[k];
					Float N_nk = doc.getSumWordWeight() + this->alphas[k] * (this->K + 1) - N_k;
					Float u1 = rg.uniform_real(), u2 = rg.uniform_real();
					Float max_uk = epsilon + pow(u1, (Float)1 / N_k)  * (pbeta[k] - epsilon);
					Float min_unk = (1 - pow(u2, (Float)1 / N_nk))
						* (1 - pbeta[k]) + pbeta[k];

					Float c = betaESum * (1 - pbeta[k]);
					lowerBound[k] = log(c * max_uk / (1 - max_uk));
					lowerBound[k] = std::max(std::min(lowerBound[k], (Float)100), (Float)-100);
					upperBound[k] = log(c * min_unk / (1 - min_unk + epsilon));
					upperBound[k] = std::max(std::min(upperBound[k], (Float)100), (Float)-100);
					if (lowerBound[k] > upperBound[k])
					{
						THROW_ERROR_WITH_INFO(exc::TrainingError,
							text::format("Bound Error: LB(%f) > UB(%f)\n"
								"max_uk: %f, min_unk: %f, c: %f", lowerBound[k], upperBound[k], max_uk, min_unk, c));
					}
				}

				try
				{
					math::sampleFromTruncatedMultiNormal(doc.beta.col((i + 1) % numBetaSample),
						topicPrior, lowerBound, upperBound, rg, numTMNSample);

					if (!std::isfinite(doc.beta.col((i + 1) % numBetaSample)[0])) 
						THROW_ERROR_WITH_INFO(exc::TrainingError,
							text::format("doc.beta.col(%d) is %f", (i + 1) % numBetaSample, 
							doc.beta.col((i + 1) % numBetaSample)[0]));
				}
				catch (const std::runtime_error& e)
				{
					std::cerr << e.what() << std::endl;
					THROW_ERROR_WITH_INFO(exc::TrainingError, e.what());
				}
			}

			// update softmax-applied beta coefficient
			doc.smBeta.head(this->K) = doc.beta.block(0, 0, this->K, std::min(numBetaSample, numDocBetaSample)).rowwise().mean();
			doc.smBeta = doc.smBeta.array().exp();
			doc.smBeta /= doc.smBeta.array().sum();
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			BaseClass::template sampleDocument<_ps, _infer>(doc, edd, docId, ld, rgs, iterationCnt, partitionId);
			/*if (iterationCnt >= this->burnIn && this->optimInterval && (iterationCnt + 1) % this->optimInterval == 0)
			{
				updateBeta(doc, rgs);
			}*/
		}

		template<GlobalSampler _gs, typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState*, _RandGen* rgs, _DocIter first, _DocIter last) const
		{
			if (this->globalStep < this->burnIn || !this->optimInterval || (this->globalStep + 1) % this->optimInterval != 0) return;

			if (pool && pool->getNumWorkers() > 1)
			{
				std::vector<std::future<void>> res;
				const size_t chStride = pool->getNumWorkers() * 8;
				size_t dist = std::distance(first, last);
				for (size_t ch = 0; ch < chStride; ++ch)
				{
					auto b = first, e = first;
					std::advance(b, dist * ch / chStride);
					std::advance(e, dist * (ch + 1) / chStride);
					res.emplace_back(pool->enqueue([&, ch, chStride](size_t threadId, _DocIter b, _DocIter e)
					{
						for (auto doc = b; doc != e; ++doc)
						{
							updateBeta(*doc, rgs[threadId]);
						}
					}, b, e));
				}
				for (auto& r : res) r.get();
			}
			else
			{
				for (auto doc = first; doc != last; ++doc)
				{
					updateBeta(*doc, rgs[0]);
				}
			}
		}

		int restoreFromTrainingError(const exc::TrainingError& e, ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			std::cerr << "Failed to sample! Reset prior and retry!" << std::endl;
			const size_t chStride = std::min(pool.getNumWorkers() * 8, this->docs.size());
			topicPrior = math::MultiNormalDistribution<Float>{ this->K };
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
			for (auto& r : res) r.get();
			return 0;
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			std::vector<std::future<void>> res;
			topicPrior = math::MultiNormalDistribution<Float>::estimate([this](size_t i)
			{
				return this->docs[i / numBetaSample].beta.col(i % numBetaSample);
			}, this->docs.size() * numBetaSample);
			if (!std::isfinite(topicPrior.mean[0]))
				THROW_ERROR_WITH_INFO(exc::TrainingError, 
					text::format("topicPrior.mean is %f", topicPrior.mean[0]));
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;

			double ll = 0;
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				Vector pbeta = doc.smBeta.array().log();
				Float last = pbeta[K - 1];
				for (Tid k = 0; k < K; ++k)
				{
					ll += pbeta[k] * (doc.numByTopic[k] + this->alphas[k]) - math::lgammaT(doc.numByTopic[k] + this->alphas[k] + 1);
				}
				pbeta.array() -= last;
				ll += topicPrior.getLL(pbeta.head(this->K));
				ll += math::lgammaT(doc.getSumWordWeight() + this->alphas.sum() + 1);
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);
			doc.beta = Matrix::Zero(this->K, numBetaSample);
			doc.smBeta = Vector::Constant(this->K, (Float)1 / this->K);
		}

		void updateDocs()
		{
			BaseClass::updateDocs();
			for (auto& doc : this->docs)
			{
				doc.beta = Matrix::Zero(this->K, numBetaSample);
			}
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			if (initDocs)
			{
				topicPrior = math::MultiNormalDistribution<Float>{ this->K };
			}
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, numBetaSample, numTMNSample, topicPrior);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, numBetaSample, numTMNSample, topicPrior);

		CTModel(const CTArgs& args)
			: BaseClass(args)
		{
			this->optimInterval = 2;
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
			if (normalize)
			{
				m = (doc.numByTopic.array().template cast<Float>() + this->alphas.array()) / (doc.getSumWordWeight() + this->alphas.sum());
			}
			else
			{
				m = doc.numByTopic.array().template cast<Float>() + this->alphas.array();
			}
			return ret;
		}

		std::vector<Float> getPriorMean() const override
		{
			return { topicPrior.mean.data(), topicPrior.mean.data() + topicPrior.mean.size() };
		}

		std::vector<Float> getPriorCov() const override
		{
			return { topicPrior.cov.data(), topicPrior.cov.data() + topicPrior.cov.size() };
		}

		std::vector<Float> getCorrelationTopic(Tid k) const override
		{
			Vector ret = topicPrior.cov.col(k).array() / (topicPrior.cov.diagonal().array() * topicPrior.cov(k, k)).sqrt();
			return { ret.data(), ret.data() + ret.size() };
		}

		GETTER(NumBetaSample, size_t, numBetaSample);

		void setNumBetaSample(size_t _numSample) override
		{
			numBetaSample = _numSample;
		}

		GETTER(NumDocBetaSample, size_t, numDocBetaSample);

		void setNumDocBetaSample(size_t _numSample) override
		{
			numDocBetaSample = _numSample;
		}

		GETTER(NumTMNSample, size_t, numTMNSample);

		void setNumTMNSample(size_t _numSample) override
		{
			numTMNSample = _numSample;
		}
	};
}
