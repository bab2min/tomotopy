#pragma once
#include "DMRModel.hpp"
#include "../Utils/slp.hpp"
#include "GDMR.h"

namespace tomoto
{
	template<TermWeight _TW>
	struct ModelStateGDMR : public ModelStateDMR<_TW>
	{
		Eigen::Matrix<FLOAT, -1, 1> alphas;
		Eigen::Matrix<FLOAT, -1, 1> terms;
		std::vector<std::vector<FLOAT>> slpCache;
		std::vector<size_t> ndimCnt;
	};

	template<TermWeight _TW, size_t _Flags = 0,
		typename _Interface = IGDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentGDMR<_TW, _Flags>,
		typename _ModelState = ModelStateGDMR<_TW>>
	class GDMRModel : public DMRModel<_TW, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_TW>, _Derived>::type;
		using BaseClass = DMRModel<_TW, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		friend typename BaseClass::BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		FLOAT sigma0 = 3;
		std::vector<FLOAT> mdCoefs, mdIntercepts;
		std::vector<size_t> degreeByF;

		FLOAT getIntegratedLambdaSq(const Eigen::Ref<const Eigen::Matrix<FLOAT, -1, 1>, 0, Eigen::InnerStride<>>& lambdas) const
		{
			FLOAT ret = pow(lambdas[0] - log(this->alpha), 2) / 2 / pow(this->sigma0, 2);
			for (size_t i = 1; i < this->F; ++i)
			{
				ret += pow(lambdas[i], 2) / 2 / pow(this->sigma, 2);
			}
			return ret;
		}

		void getIntegratedLambdaSqP(const Eigen::Ref<const Eigen::Matrix<FLOAT, -1, 1>, 0, Eigen::InnerStride<>>& lambdas, 
			Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>, 0, Eigen::InnerStride<>> ret) const
		{
			ret[0] = (lambdas[0] - log(this->alpha)) / pow(this->sigma0, 2);
			for (size_t i = 1; i < this->F; ++i)
			{
				ret[i] = lambdas[i] / pow(this->sigma, 2);
			}
		}

		void initParameters()
		{
			auto dist0 = std::normal_distribution<FLOAT>(log(this->alpha), sigma0);
			auto dist = std::normal_distribution<FLOAT>(0, this->sigma);
			for (size_t i = 0; i < this->K; ++i) for (size_t j = 0; j < this->F; ++j)
			{
				if (j == 0)
				{
					this->lambda(i, j) = dist0(this->rg);
				}
				else
				{
					this->lambda(i, j) = dist(this->rg);
				}
			}
		}

		FLOAT getNegativeLambdaLL(Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g) const
		{
			auto mappedX = Eigen::Map<Eigen::Matrix<FLOAT, -1, -1>>(x.data(), this->K, this->F);
			auto mappedG = Eigen::Map<Eigen::Matrix<FLOAT, -1, -1>>(g.data(), this->K, this->F);

			FLOAT fx = 0;
			for (size_t k = 0; k < this->K; ++k)
			{
				fx += getIntegratedLambdaSq(mappedX.row(k));
				getIntegratedLambdaSqP(mappedX.row(k), mappedG.row(k));
			}
			return fx;
		}

		FLOAT evaluateLambdaObj(Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> x, Eigen::Matrix<FLOAT, -1, 1>& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing search more
			if ((x.array() > this->maxLambda).any()) return INFINITY;

			const auto K = this->K;
			const auto F = this->F;

			auto mappedX = Eigen::Map<Eigen::Matrix<FLOAT, -1, -1>>(x.data(), K, F);
			FLOAT fx = -static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);

			std::vector<std::future<Eigen::Matrix<FLOAT, -1, 1>>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this](size_t threadId)
				{
					auto& ld = localData[threadId];
					auto& alphas = ld.alphas;
					auto& tmpK = ld.tmpK;
					auto& terms = ld.terms;
					Eigen::Matrix<FLOAT, -1, 1> ret = Eigen::Matrix<FLOAT, -1, 1>::Zero(F * K + 1);
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						const auto& vx = doc.metadataC;
						getTermsFromMd(ld, &vx[0], terms);
						for (TID k = 0; k < K; ++k)
						{
							alphas[k] = exp(mappedX.row(k) * terms) + this->alphaEps;
							ret[K * F] -= math::lgammaT(alphas[k]) - math::lgammaT(doc.numByTopic[k] + alphas[k]);
							if (!std::isfinite(alphas[k]) && alphas[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphas[k]) - math::digammaT(doc.numByTopic[k] + alphas[k]));
						}
						FLOAT alphaSum = alphas.sum();
						ret[K * F] += math::lgammaT(alphaSum) - math::lgammaT(doc.getSumWordWeight() + alphaSum);
						FLOAT t = math::digammaT(alphaSum) - math::digammaT(doc.getSumWordWeight() + alphaSum);
						if (!std::isfinite(alphaSum) && alphaSum > 0)
						{
							ret[K * F] = -INFINITY;
							t = 0;
						}
						for (size_t f = 0; f < F; ++f)
						{
							ret.segment(f * K, K).array() -= ((tmpK.array() + t) * alphas.array()) * terms[f];
						}
					}
					return ret;
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

		void getTermsFromMd(_ModelState& ld, const FLOAT* vx, Eigen::Ref<Eigen::Matrix<FLOAT, -1, 1>> out) const
		{
			auto& digit = ld.ndimCnt;
			std::fill(digit.begin(), digit.end(), 0);
			for (size_t n = 0; n < degreeByF.size(); ++n)
			{
				for (size_t i = 0; i < degreeByF[n]; ++i)
				{
					ld.slpCache[n][i] = slp::slpGet(i + 1, vx[n]);
				}
			}

			for (size_t i = 0; i < this->F; ++i)
			{
				out[i] = 1;
				for (size_t n = 0; n < degreeByF.size(); ++n)
				{
					if(digit[n]) out[i] *= ld.slpCache[n][digit[n] - 1];
				}

				size_t u;
				for (u = 0; u < digit.size() && ++digit[u] > degreeByF[u]; ++u)
				{
					digit[u] = 0;
				}
				u = std::min(u, degreeByF.size() - 1);
			}
		}

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			getTermsFromMd(ld, &doc.metadataC[0], ld.terms);
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + (this->lambda * ld.terms).array().exp() + this->alphaEps)
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			double ll = 0;

			Eigen::Matrix<FLOAT, -1, 1> alphas(K);
			auto tempState = this->globalState;
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				auto& terms = tempState.terms;
				getTermsFromMd(tempState, &doc.metadataC[0], terms);
				for (TID k = 0; k < K; ++k)
				{
					alphas[k] = exp(this->lambda.row(k) * terms) + this->alphaEps;
				}
				FLOAT alphaSum = alphas.sum();
				for (TID k = 0; k < K; ++k)
				{
					if (!doc.numByTopic[k]) continue;
					ll += math::lgammaT(doc.numByTopic[k] + alphas[k]) - math::lgammaT(alphas[k]);
				}
				ll -= math::lgammaT(doc.getSumWordWeight() + alphaSum) - math::lgammaT(alphaSum);
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			const size_t V = this->realV;
			const auto K = this->K;
			const auto eta = this->eta;
			double ll = 0;
			for (size_t k = 0; k < K; ++k)
			{
				ll += getIntegratedLambdaSq(this->lambda.row(k));
			}
			ll /= -2 * pow(this->sigma, 2);

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

		void normalizeMetadata()
		{
			size_t s = degreeByF.size();
			if (mdIntercepts.size() < s || mdCoefs.size() < s)
			{
				mdIntercepts.resize(s, FLT_MAX);
				mdCoefs.resize(s, FLT_MIN);
			}

			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < s; ++i)
				{
					mdIntercepts[i] = std::min(mdIntercepts[i], doc.metadataC[i]);
					mdCoefs[i] = std::max(mdCoefs[i], doc.metadataC[i]);
				}
			}
			for (size_t i = 0; i < s; ++i)
			{
				mdCoefs[i] -= mdIntercepts[i];
				if (mdCoefs[i] == 0) mdCoefs[i] = 1;
			}
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
			for (size_t i = 0; i < degreeByF.size(); ++i) doc.metadataC[i] = mdCoefs[i] ? (doc.metadataC[i] - mdIntercepts[i]) / mdCoefs[i] : 0;
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::BaseClass::initGlobalState(initDocs);
			this->globalState.tmpK = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K);
			this->globalState.alphas = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K);
			this->globalState.terms = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->F);
			for (auto& f : degreeByF)
			{
				this->globalState.slpCache.emplace_back(std::vector<FLOAT>(f));
			}
			this->globalState.ndimCnt.resize(degreeByF.size());
			normalizeMetadata();
			
			if (initDocs)
			{
				this->lambda = Eigen::Matrix<FLOAT, -1, -1>::Zero(this->K, this->F);
				this->lambda.col(0).fill(log(this->alpha));
			}
			LBFGSpp::LBFGSParam<FLOAT> param;
			param.max_iterations = this->maxBFGSIteration;
			this->solver = decltype(this->solver){ param };
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, sigma0, degreeByF, mdCoefs, mdIntercepts);
	public:
		GDMRModel(size_t _K = 1, const std::vector<size_t>& _degreeByF = {}, FLOAT defaultAlpha = 1.0, FLOAT _sigma = 1.0, FLOAT _eta = 0.01,
			FLOAT _alphaEps = 1e-10, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, defaultAlpha, _sigma, _eta, _alphaEps, _rg), degreeByF(_degreeByF)
		{
			this->F = accumulate(degreeByF.begin(), degreeByF.end(), 1, [](size_t a, size_t b) {return a * (b + 1); });
		}

		GETTER(Fs, const std::vector<size_t>&, degreeByF);
		GETTER(Sigma0, FLOAT, sigma0);

		void setSigma0(FLOAT _sigma0)
		{
			this->sigma0 = _sigma0;
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) override
		{
			auto doc = this->_makeDoc(words);
			transform(metadata.begin(), metadata.end(), back_inserter(doc.metadataC), [](const std::string& s)
			{
				return stof(s);
			});
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const override
		{
			auto doc = this->_makeDocWithinVocab(words);
			std::transform(metadata.begin(), metadata.end(), back_inserter(doc.metadataC), [](const std::string& s)
			{
				return stof(s);
			});
			return make_unique<_DocType>(doc);
		}

		std::vector<FLOAT> getTopicsByDoc(const _DocType& doc) const
		{
			Eigen::Matrix<FLOAT, -1, 1> alphas(this->K);
			auto tempState = this->globalState;
			auto& terms = tempState.terms;
			getTermsFromMd(tempState, &doc.metadataC[0], terms);
			for (TID k = 0; k < this->K; ++k)
			{
				alphas[k] = exp(this->lambda.row(k) * terms) + this->alphaEps;
			}
			std::vector<FLOAT> ret(this->K);
			FLOAT sum = doc.getSumWordWeight() + alphas.sum();
			for (size_t k = 0; k < this->K; ++k)
			{
				ret[k] = (doc.numByTopic[k] + alphas[k]) / sum;
			}
			return ret;
		}

		std::vector<FLOAT> getLambdaByTopic(TID tid) const override
		{
			std::vector<FLOAT> ret(this->F);
			for (size_t f = 0; f < this->F; ++f) ret[f] = this->lambda.row(tid)[f];
			return ret;
		}

		void setMdRange(const std::vector<FLOAT>& vMin, const std::vector<FLOAT>& vMax) override
		{
			mdIntercepts = vMin;
			mdCoefs = vMax;
		}
	};
}