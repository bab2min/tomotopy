#pragma once
#include "LDAModel.hpp"
#include "../Utils/LBFGS.h"
#include "../Utils/text.hpp"
#include "DMR.h"
/*
Implementation of DMR using Gibbs sampling by bab2min
* Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.
*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateDMR : public ModelStateLDA<_tw>
	{
		Eigen::Matrix<Float, -1, 1> tmpK;
	};

	template<TermWeight _tw, size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentDMR<_tw>,
		typename _ModelState = ModelStateDMR<_tw>>
	class DMRModel : public LDAModel<_tw, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_tw, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr char TMID[] = "DMR\0";

		Eigen::Matrix<Float, -1, -1> lambda;
		Eigen::Matrix<Float, -1, -1> expLambda;
		Float sigma;
		size_t F = 0;
		size_t optimRepeat = 5;
		Float alphaEps = 1e-10;
		Float temperatureScale = 0;
		static constexpr Float maxLambda = 10;
		static constexpr size_t maxBFGSIteration = 10;

		Dictionary metadataDict;
		LBFGSpp::LBFGSSolver<Float, LBFGSpp::LineSearchBracketing> solver;

		Float getNegativeLambdaLL(Eigen::Ref<Eigen::Matrix<Float, -1, 1>> x, Eigen::Matrix<Float, -1, 1>& g) const
		{
			g = (x.array() - log(this->alpha)) / pow(sigma, 2);
			return (x.array() - log(this->alpha)).pow(2).sum() / 2 / pow(sigma, 2);
		}

		Float evaluateLambdaObj(Eigen::Ref<Eigen::Matrix<Float, -1, 1>> x, Eigen::Matrix<Float, -1, 1>& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing searching more
			if ((x.array() > maxLambda).any()) return INFINITY;

			const auto K = this->K;

			Float fx = - static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);
			auto alphas = (x.array().exp() + alphaEps).eval();

			std::vector<std::future<Eigen::Matrix<Float, -1, 1>>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&](size_t threadId)
				{
					auto& tmpK = localData[threadId].tmpK;
					if (!tmpK.size()) tmpK.resize(this->K);
					Eigen::Matrix<Float, -1, 1> val = Eigen::Matrix<Float, -1, 1>::Zero(K * F + 1);
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						auto alphaDoc = alphas.segment(doc.metadata * K, K);
						Float alphaSum = alphaDoc.sum();
						for (Tid k = 0; k < K; ++k)
						{
							val[K * F] -= math::lgammaT(alphaDoc[k]) - math::lgammaT(doc.numByTopic[k] + alphaDoc[k]);
							if (!std::isfinite(alphaDoc[k]) && alphaDoc[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphaDoc[k]) - math::digammaT(doc.numByTopic[k] + alphaDoc[k]));
						}
						//val[K * F] = -(lgammaApprox(alphaDoc.array()) - lgammaApprox(doc.numByTopic.array().cast<Float>() + alphaDoc.array())).sum();
						//tmpK = -(digammaApprox(alphaDoc.array()) - digammaApprox(doc.numByTopic.array().cast<Float>() + alphaDoc.array()));
						val[K * F] += math::lgammaT(alphaSum) - math::lgammaT(doc.getSumWordWeight() + alphaSum);
						Float t = math::digammaT(alphaSum) - math::digammaT(doc.getSumWordWeight() + alphaSum);
						if (!std::isfinite(alphaSum) && alphaSum > 0)
						{
							val[K * F] = -INFINITY;
							t = 0;
						}
						val.segment(doc.metadata * K, K).array() -= alphaDoc.array() * (tmpK.array() + t);
					}
					return val;
				}));
			}
			for (auto& r : res)
			{
				auto ret = r.get();
				fx += ret[K * F];
				g += ret.head(K * F);
			}

			// positive fx is an error from limited precision of float.
			if (fx > 0) return INFINITY;
			return -fx;
		}

		void initParameters()
		{
			auto dist = std::normal_distribution<Float>(log(this->alpha), sigma);
			for (size_t i = 0; i < this->K; ++i) for (size_t j = 0; j < F; ++j)
			{
				lambda(i, j) = dist(this->rg);
			}
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			Eigen::Matrix<Float, -1, -1> bLambda;
			Float fx = 0, bestFx = INFINITY;
			for (size_t i = 0; i < optimRepeat; ++i)
			{
				static_cast<DerivedClass*>(this)->initParameters();
				int ret = solver.minimize([this, &pool, localData](Eigen::Ref<Eigen::Matrix<Float, -1, 1>> x, Eigen::Matrix<Float, -1, 1>& g)
				{
					return static_cast<DerivedClass*>(this)->evaluateLambdaObj(x, g, pool, localData);
				}, Eigen::Map<Eigen::Matrix<Float, -1, 1>>(lambda.data(), lambda.size()), fx);

				if (fx < bestFx)
				{
					bLambda = lambda;
					bestFx = fx;
					//printf("\t(%d) %e\n", ret, fx);
				}
			}
			if (!std::isfinite(bestFx))
			{
				throw exception::TrainingError{ "optimizing parameters has been failed!" };
			}
			lambda = bLambda;
			//std::cerr << fx << std::endl;
			expLambda = lambda.array().exp() + alphaEps;
		}

		int restoreFromTrainingError(const exception::TrainingError& e, ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			std::cerr << "Failed to optimize! Reset prior and retry!" << std::endl;
			lambda.setZero();
			expLambda = lambda.array().exp() + alphaEps;
			return 0;
		}

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + this->expLambda.col(doc.metadata).array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		
		double getLLDocTopic(const _DocType& doc) const
		{
			const size_t V = this->realV;
			const auto K = this->K;

			auto alphaDoc = expLambda.col(doc.metadata);

			Float ll = 0;
			Float alphaSum = alphaDoc.sum();
			for (Tid k = 0; k < K; ++k)
			{
				ll += math::lgammaT(doc.numByTopic[k] + alphaDoc[k]);
				ll -= math::lgammaT(alphaDoc[k]);
			}
			ll -= math::lgammaT(doc.getSumWordWeight() + alphaSum);
			ll += math::lgammaT(alphaSum);
			return ll;
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;

			double ll = 0;
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				auto alphaDoc = expLambda.col(doc.metadata);
				Float alphaSum = alphaDoc.sum();

				for (Tid k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alphaDoc[k]) - math::lgammaT(alphaDoc[k]);
				}
				ll -= math::lgammaT(doc.getSumWordWeight() + alphaSum) - math::lgammaT(alphaSum);
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			const size_t V = this->realV;

			double ll = -(lambda.array() - log(alpha)).pow(2).sum() / 2 / pow(sigma, 2);
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
				}
			}
			return ll;
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			this->globalState.tmpK = Eigen::Matrix<Float, -1, 1>::Zero(this->K);
			F = metadataDict.size();
			if (initDocs)
			{
				lambda = Eigen::Matrix<Float, -1, -1>::Constant(this->K, F, log(this->alpha));
			}
			if (_Flags & flags::continuous_doc_data) this->numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, this->docs.size());
			expLambda = lambda.array().exp();
			LBFGSpp::LBFGSParam<Float> param;
			param.max_iterations = maxBFGSIteration;
			solver = decltype(solver){ param };
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, sigma, alphaEps, metadataDict, lambda);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, sigma, alphaEps, metadataDict, lambda);

		DMRModel(size_t _K = 1, Float defaultAlpha = 1.0, Float _sigma = 1.0, Float _eta = 0.01, 
			Float _alphaEps = 0, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, defaultAlpha, _eta, _rg), sigma(_sigma), alphaEps(_alphaEps)
		{
			if (_sigma <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong sigma value (sigma = %f)", _sigma));
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<std::string>& metadata)
		{
			std::string metadataJoined = text::join(metadata.begin(), metadata.end(), "_");
			Vid xid;
			if (_const)
			{
				xid = metadataDict.toWid(metadataJoined);
				if (xid == (Vid)-1) throw std::invalid_argument("unknown metadata");
			}
			else
			{
				xid = metadataDict.add(metadataJoined);
			}
			doc.metadata = xid;
			return doc;
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) override
		{
			auto doc = this->_makeDoc(words);
			return this->_addDoc(_updateDoc(doc, metadata));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const override
		{
			auto doc = as_mutable(this)->template _makeDoc<true>(words);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, metadata));
		}

		size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer, 
			const std::vector<std::string>& metadata) override
		{
			auto doc = this->template _makeRawDoc<false>(rawStr, tokenizer);
			return this->_addDoc(_updateDoc(doc, metadata));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer, 
			const std::vector<std::string>& metadata) const override
		{
			auto doc = as_mutable(this)->template _makeRawDoc<true>(rawStr, tokenizer);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, metadata));
		}

		size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len, 
			const std::vector<std::string>& metadata) override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return this->_addDoc(_updateDoc(doc, metadata));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len, 
			const std::vector<std::string>& metadata) const override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, metadata));
		}

		GETTER(F, size_t, F);
		GETTER(Sigma, Float, sigma);
		GETTER(AlphaEps, Float, alphaEps);
		GETTER(OptimRepeat, size_t, optimRepeat);

		void setAlphaEps(Float _alphaEps) override
		{
			alphaEps = _alphaEps;
		}

		void setOptimRepeat(size_t _optimRepeat) override
		{
			optimRepeat = _optimRepeat;
		}

		std::vector<Float> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<Float> ret(this->K);
			auto alphaDoc = expLambda.col(doc.metadata);
			Eigen::Map<Eigen::Matrix<Float, -1, 1>>{ret.data(), this->K}.array() =
				(doc.numByTopic.array().template cast<Float>() + alphaDoc.array()) / (doc.getSumWordWeight() + alphaDoc.sum());
			return ret;
		}

		std::vector<Float> getLambdaByMetadata(size_t metadataId) const override
		{
			assert(metadataId < metadataDict.size());
			auto l = lambda.col(metadataId);
			return { l.data(), l.data() + this->K };
		}

		std::vector<Float> getLambdaByTopic(Tid tid) const override
		{
			assert(tid < this->K);
			auto l = lambda.row(tid);
			return { l.data(), l.data() + F };
		}

		const Dictionary& getMetadataDict() const override { return metadataDict; }
	};

	/* This is for preventing 'undefined symbol' problem in compiling by clang. */
	template<TermWeight _tw, size_t _Flags,
		typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
		constexpr Float DMRModel<_tw, _Flags, _Interface, _Derived, _DocType, _ModelState>::maxLambda;

	template<TermWeight _tw, size_t _Flags,
		typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
		constexpr size_t DMRModel<_tw, _Flags, _Interface, _Derived, _DocType, _ModelState>::maxBFGSIteration;
}
