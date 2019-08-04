#pragma once
#include "LDAModel.hpp"
#include "../Utils/LBFGS.h"
#include "../Utils/text.hpp"

/*
Implementation of DMR using Gibbs sampling by bab2min
* Mimno, D., & McCallum, A. (2012). Topic models conditioned on arbitrary features with dirichlet-multinomial regression. arXiv preprint arXiv:1206.3278.
*/

namespace tomoto
{
	template<TermWeight _TW, bool _Shared = false>
	struct DocumentDMR : public DocumentLDA<_TW, _Shared>
	{
		using DocumentLDA<_TW, _Shared>::DocumentLDA;
		size_t metadata = 0;

		DEFINE_SERIALIZER_AFTER_BASE2(DocumentLDA<_TW, _Shared>, metadata);
	};

	template<TermWeight _TW>
	struct ModelStateDMR : public ModelStateLDA<_TW>
	{
		Eigen::Matrix<FLOAT, -1, 1> tmpK;
	};

	class IDMRModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IDMRModel* create(TermWeight _weight, size_t _K = 1,
			FLOAT defaultAlpha = 1.0, FLOAT _sigma = 1.0, FLOAT _eta = 0.01, FLOAT _alphaEps = 1e-10,
			const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const = 0;
		
		virtual void setAlphaEps(FLOAT _alphaEps) = 0;
		virtual FLOAT getAlphaEps() const = 0;
		virtual void setOptimRepeat(size_t repeat) = 0;
		virtual size_t getOptimRepeat() const = 0;
		virtual size_t getF() const = 0;
		virtual FLOAT getSigma() const = 0;
		virtual const Dictionary& getMetadataDict() const = 0;
		virtual std::vector<FLOAT> getLambdaByMetadata(size_t metadataId) const = 0;
		virtual std::vector<FLOAT> getLambdaByTopic(TID tid) const = 0;
	};

	template<TermWeight _TW, bool _Shared = false,
		typename _Interface = IDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentDMR<_TW>,
		typename _ModelState = ModelStateDMR<_TW>>
	class DMRModel : public LDAModel<_TW, _Shared, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_TW, _Shared>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "DMR";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, _Shared, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Eigen::Matrix<FLOAT, -1, -1> lambda;
		Eigen::Matrix<FLOAT, -1, -1> expLambda;
		FLOAT sigma;
		size_t F = 0;
		size_t optimRepeat = 5;
		FLOAT alphaEps = 1e-10;
		FLOAT temperatureScale = 0;
		static constexpr FLOAT maxLambda = 10;
		static constexpr size_t maxBFGSIteration = 10;

		Dictionary metadataDict;
		LBFGSpp::LBFGSSolver<FLOAT, LBFGSpp::LineSearchBracketing> solver;

		FLOAT getNegativeLambdaLL(Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g) const
		{
			g = (x.array() - log(this->alpha)) / pow(sigma, 2);
			return (x.array() - log(this->alpha)).pow(2).sum() / 2 / pow(sigma, 2);
		}

		FLOAT evaluateLambdaObj(Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing searching more
			if ((x.array() > maxLambda).any()) return INFINITY;

			const auto K = this->K;

			FLOAT fx = - static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);
			auto alphas = (x.array().exp() + alphaEps).eval();

			std::vector<std::future<Eigen::Matrix<FLOAT, -1, 1>>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&](size_t threadId)
				{
					auto& tmpK = localData[threadId].tmpK;
					Eigen::Matrix<FLOAT, -1, 1> val = Eigen::Matrix<FLOAT, -1, 1>::Zero(K * F + 1);
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						auto alphaDoc = alphas.segment(doc.metadata * K, K);
						FLOAT alphaSum = alphaDoc.sum();
						for (TID k = 0; k < K; ++k)
						{
							val[K * F] -= math::lgammaT(alphaDoc[k]) - math::lgammaT(doc.numByTopic[k] + alphaDoc[k]);
							if (!isfinite(alphaDoc[k]) && alphaDoc[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphaDoc[k]) - math::digammaT(doc.numByTopic[k] + alphaDoc[k]));
						}
						//val[K * F] = -(lgammaApprox(alphaDoc.array()) - lgammaApprox(doc.numByTopic.array().cast<FLOAT>() + alphaDoc.array())).sum();
						//tmpK = -(digammaApprox(alphaDoc.array()) - digammaApprox(doc.numByTopic.array().cast<FLOAT>() + alphaDoc.array()));
						val[K * F] += math::lgammaT(alphaSum) - math::lgammaT(doc.template getSumWordWeight<_TW>() + alphaSum);
						FLOAT t = math::digammaT(alphaSum) - math::digammaT(doc.template getSumWordWeight<_TW>() + alphaSum);
						if (!isfinite(alphaSum) && alphaSum > 0)
						{
							val[K * F] = -INFINITY;
							t = 0;
						}
						val.segment(doc.metadata * K, K).array() -= alphaDoc.array() * (tmpK.array() + t);
					}
					return val;
				}));
			}
			for (auto&& r : res)
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
			auto dist = std::normal_distribution<FLOAT>(log(this->alpha), sigma);
			for (size_t i = 0; i < this->K; ++i) for (size_t j = 0; j < F; ++j)
			{
				lambda(i, j) = dist(this->rg);
			}
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData)
		{
			Eigen::Matrix<FLOAT, -1, -1> bLambda;
			FLOAT fx = 0, bestFx = INFINITY;
			for (size_t i = 0; i < optimRepeat; ++i)
			{
				static_cast<DerivedClass*>(this)->initParameters();
				int ret = solver.minimize([this, &pool, localData](Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g)
				{
					return static_cast<DerivedClass*>(this)->evaluateLambdaObj(x, g, pool, localData);
				}, Eigen::Map<Eigen::Matrix<FLOAT, -1, 1>>(lambda.data(), lambda.size()), fx);

				if (fx < bestFx)
				{
					bLambda = lambda;
					bestFx = fx;
					//printf("\t(%d) %e\n", ret, fx);
				}
			}
			if (!std::isfinite(bestFx))
			{
				std::cout << "optimizing parameters has been failed!" << std::endl;
				throw std::runtime_error{ "optimizing parameters has been failed!" };
			}
			lambda = bLambda;
			//std::cerr << fx << std::endl;
			expLambda = lambda.array().exp() + alphaEps;
		}

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t vid) const
		{
			const size_t V = this->realV;
			const auto K = this->K;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + this->expLambda.col(doc.metadata).array())
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		
		double getLLDocTopic(const _DocType& doc) const
		{
			const size_t V = this->realV;
			const auto K = this->K;

			auto alphaDoc = expLambda.segment(doc.metadata * K, K);

			FLOAT ll = 0;
			FLOAT alphaSum = alphaDoc.sum();
			for (TID k = 0; k < K; ++k)
			{
				ll += math::lgammaT(doc.numByTopic[k] + alphaDoc[k]);
				ll -= math::lgammaT(alphaDoc[k]);
			}
			ll -= math::lgammaT(doc.template getSumWordWeight<_TW>() + alphaSum);
			ll += math::lgammaT(alphaSum);
			return ll;
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
				auto alphaDoc = expLambda.col(doc.metadata);
				FLOAT alphaSum = alphaDoc.sum();

				for (TID k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alphaDoc[k]) - math::lgammaT(alphaDoc[k]);
				}
				ll -= math::lgammaT(doc.template getSumWordWeight<_TW>() + alphaSum) - math::lgammaT(alphaSum);
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

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			this->globalState.tmpK = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K);
			F = metadataDict.size();
			if (initDocs)
			{
				lambda = Eigen::Matrix<FLOAT, -1, -1>::Constant(this->K, F, log(this->alpha));
			}
			if (_Shared) this->numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, this->docs.size());
			expLambda = lambda.array().exp();
			LBFGSpp::LBFGSParam<FLOAT> param;
			param.max_iterations = maxBFGSIteration;
			solver = decltype(solver){ param };
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, sigma, alphaEps, metadataDict, lambda);

	public:
		DMRModel(size_t _K = 1, FLOAT defaultAlpha = 1.0, FLOAT _sigma = 1.0, FLOAT _eta = 0.01, 
			FLOAT _alphaEps = 0, const RANDGEN& _rg = RANDGEN{ std::random_device{}() })
			: BaseClass(_K, defaultAlpha, _eta, _rg), sigma(_sigma), alphaEps(_alphaEps)
		{
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) override
		{
			std::string metadataJoined = text::join(metadata.begin(), metadata.end(), "_");
			VID xid = metadataDict.add(metadataJoined);
			auto doc = this->_makeDoc(words);
			doc.metadata = xid;
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const override
		{
			std::string metadataJoined = text::join(metadata.begin(), metadata.end(), "_");
			VID xid = metadataDict.toWid(metadataJoined);
			if (xid == (VID)-1) throw std::invalid_argument("unknown metadata");
			auto doc = this->_makeDocWithinVocab(words);
			doc.metadata = xid;
			return std::make_unique<_DocType>(doc);
		}

		GETTER(F, size_t, F);
		GETTER(Sigma, FLOAT, sigma);
		GETTER(AlphaEps, FLOAT, alphaEps);
		GETTER(OptimRepeat, size_t, optimRepeat);

		void setAlphaEps(FLOAT _alphaEps)
		{
			alphaEps = _alphaEps;
		}

		void setOptimRepeat(size_t _optimRepeat)
		{
			optimRepeat = _optimRepeat;
		}

		std::vector<FLOAT> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<FLOAT> ret(this->K);
			auto alphaDoc = expLambda.col(doc.metadata);
			FLOAT sum = doc.template getSumWordWeight<_TW>() + alphaDoc.sum();
			for (size_t k = 0; k < this->K; ++k)
			{
				ret[k] = (doc.numByTopic[k] + alphaDoc[k]) / sum;
			}
			return ret;
		}

		std::vector<FLOAT> getLambdaByMetadata(size_t metadataId) const override
		{
			assert(metadataId < metadataDict.size());
			auto l = lambda.col(metadataId);
			return { l.data(), l.data() + this->K };
		}

		std::vector<FLOAT> getLambdaByTopic(TID tid) const override
		{
			assert(tid < this->K);
			auto l = lambda.row(tid);
			return { l.data(), l.data() + F };
		}

		const Dictionary& getMetadataDict() const { return metadataDict; }
	};

	IDMRModel* IDMRModel::create(TermWeight _weight, size_t _K, FLOAT _defaultAlpha, FLOAT _sigma, FLOAT _eta, FLOAT _alphaEps, const RANDGEN& _rg)
	{
		SWITCH_TW(_weight, DMRModel, _K, _defaultAlpha, _sigma, _eta, _alphaEps, _rg);
	}
}
