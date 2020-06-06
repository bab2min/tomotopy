#pragma once
#include "DMRModel.hpp"
#include "../Utils/slp.hpp"
#include "GDMR.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateGDMR : public ModelStateDMR<_tw>
	{
		/*Eigen::Matrix<Float, -1, 1> alphas;
		Eigen::Matrix<Float, -1, 1> terms;
		std::vector<std::vector<Float>> slpCache;
		std::vector<size_t> ndimCnt;*/
	};

	template<TermWeight _tw, size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IGDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentGDMR<_tw, _Flags>,
		typename _ModelState = ModelStateGDMR<_tw>>
	class GDMRModel : public DMRModel<_tw, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_tw>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_tw>, _Derived>::type;
		using BaseClass = DMRModel<_tw, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		friend typename BaseClass::BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Float sigma0 = 3;
		std::vector<Float> mdCoefs, mdIntercepts, mdMax;
		std::vector<uint64_t> degreeByF;

		Float getIntegratedLambdaSq(const Eigen::Ref<const Eigen::Matrix<Float, -1, 1>, 0, Eigen::InnerStride<>>& lambdas) const
		{
			Float ret = pow(lambdas[0] - log(this->alpha), 2) / 2 / pow(this->sigma0, 2);
			for (size_t i = 1; i < this->F; ++i)
			{
				ret += pow(lambdas[i], 2) / 2 / pow(this->sigma, 2);
			}
			return ret;
		}

		void getIntegratedLambdaSqP(const Eigen::Ref<const Eigen::Matrix<Float, -1, 1>, 0, Eigen::InnerStride<>>& lambdas, 
			Eigen::Ref<Eigen::Matrix<Float, -1, 1>, 0, Eigen::InnerStride<>> ret) const
		{
			ret[0] = (lambdas[0] - log(this->alpha)) / pow(this->sigma0, 2);
			for (size_t i = 1; i < this->F; ++i)
			{
				ret[i] = lambdas[i] / pow(this->sigma, 2);
			}
		}

		void initParameters()
		{
			auto dist0 = std::normal_distribution<Float>(log(this->alpha), sigma0);
			auto dist = std::normal_distribution<Float>(0, this->sigma);
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

		Float getNegativeLambdaLL(Eigen::Ref<Eigen::Matrix<Float, -1, 1>> x, Eigen::Matrix<Float, -1, 1>& g) const
		{
			auto mappedX = Eigen::Map<Eigen::Matrix<Float, -1, -1>>(x.data(), this->K, this->F);
			auto mappedG = Eigen::Map<Eigen::Matrix<Float, -1, -1>>(g.data(), this->K, this->F);

			Float fx = 0;
			for (size_t k = 0; k < this->K; ++k)
			{
				fx += getIntegratedLambdaSq(mappedX.row(k));
				getIntegratedLambdaSqP(mappedX.row(k), mappedG.row(k));
			}
			return fx;
		}

		Float evaluateLambdaObj(Eigen::Ref<Eigen::Matrix<Float, -1, 1>> x, Eigen::Matrix<Float, -1, 1>& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing search more
			if ((x.array() > this->maxLambda).any()) return INFINITY;

			const auto K = this->K;
			const auto F = this->F;

			auto mappedX = Eigen::Map<Eigen::Matrix<Float, -1, -1>>(x.data(), K, F);
			Float fx = -static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);

			std::vector<std::future<Eigen::Matrix<Float, -1, 1>>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this](size_t threadId)
				{
					auto& ld = localData[threadId];
					thread_local Eigen::Matrix<Float, -1, 1> alphas{ K }, tmpK{ K }, terms{ F };
					Eigen::Matrix<Float, -1, 1> ret = Eigen::Matrix<Float, -1, 1>::Zero(F * K + 1);
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						const auto& vx = doc.metadataNormalized;
						getTermsFromMd(&vx[0], terms.data());
						for (Tid k = 0; k < K; ++k)
						{
							alphas[k] = exp(mappedX.row(k) * terms) + this->alphaEps;
							ret[K * F] -= math::lgammaT(alphas[k]) - math::lgammaT(doc.numByTopic[k] + alphas[k]);
							if (!std::isfinite(alphas[k]) && alphas[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphas[k]) - math::digammaT(doc.numByTopic[k] + alphas[k]));
						}
						Float alphaSum = alphas.sum();
						ret[K * F] += math::lgammaT(alphaSum) - math::lgammaT(doc.getSumWordWeight() + alphaSum);
						Float t = math::digammaT(alphaSum) - math::digammaT(doc.getSumWordWeight() + alphaSum);
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

		void getTermsFromMd(const Float* vx, Float* out, bool normalize = false) const
		{
			thread_local std::vector<size_t> digit(degreeByF.size());
			std::fill(digit.begin(), digit.end(), 0);
			
			thread_local std::vector<std::vector<Float>> slpCache;
			if (slpCache.empty())
			{
				for (auto& f : degreeByF)
				{
					slpCache.emplace_back(std::vector<Float>(f));
				}
			}

			
			for (size_t n = 0; n < degreeByF.size(); ++n)
			{
				for (size_t i = 0; i < degreeByF[n]; ++i)
				{
					slpCache[n][i] = slp::slpGet(i + 1, normalize ? ((vx[n] - mdIntercepts[n]) / mdCoefs[n]) : vx[n]);
				}
			}

			for (size_t i = 0; i < this->F; ++i)
			{
				out[i] = 1;
				for (size_t n = 0; n < degreeByF.size(); ++n)
				{
					if(digit[n]) out[i] *= slpCache[n][digit[n] - 1];
				}

				size_t u;
				for (u = 0; u < digit.size() && ++digit[u] > degreeByF[u]; ++u)
				{
					digit[u] = 0;
				}
				u = std::min(u, degreeByF.size() - 1);
			}
		}

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			thread_local Eigen::Matrix<Float, -1, 1> terms{ this->F };
			getTermsFromMd(&doc.metadataNormalized[0], terms.data());
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + (this->lambda * terms).array().exp() + this->alphaEps)
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			double ll = 0;

			Eigen::Matrix<Float, -1, 1> alphas(K);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				thread_local Eigen::Matrix<Float, -1, 1> terms{ this->F };
				getTermsFromMd(&doc.metadataNormalized[0], terms.data());
				for (Tid k = 0; k < K; ++k)
				{
					alphas[k] = exp(this->lambda.row(k) * terms) + this->alphaEps;
				}
				Float alphaSum = alphas.sum();
				for (Tid k = 0; k < K; ++k)
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

		void collectMinMaxMetadata()
		{
			size_t s = degreeByF.size();
			if (mdIntercepts.size() < s)
			{
				mdIntercepts.resize(s, FLT_MAX);
				mdMax.resize(s, FLT_MIN);
			}
			mdCoefs.resize(s, 0);

			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < s; ++i)
				{
					mdIntercepts[i] = std::min(mdIntercepts[i], doc.metadataOrg[i]);
					mdMax[i] = std::max(mdMax[i], doc.metadataOrg[i]);
				}
			}
			for (size_t i = 0; i < s; ++i)
			{
				mdCoefs[i] = mdMax[i] - mdIntercepts[i];
				if (mdCoefs[i] == 0) mdCoefs[i] = 1;
			}
		}

		std::vector<Float> normalizeMetadata(const std::vector<Float>& metadata) const
		{
			std::vector<Float> ret(degreeByF.size());
			for (size_t i = 0; i < degreeByF.size(); ++i)
			{
				ret[i] = mdCoefs[i] ? (metadata[i] - mdIntercepts[i]) / mdCoefs[i] : 0;
			}
			return ret;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);
			doc.metadataNormalized = normalizeMetadata(doc.metadataOrg);
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::BaseClass::initGlobalState(initDocs);
			this->F = accumulate(degreeByF.begin(), degreeByF.end(), 1, [](size_t a, size_t b) {return a * (b + 1); });
			if (initDocs) collectMinMaxMetadata();
			else
			{
				// Old binary file has metadataNormalized values into `metadataOrg`
				if (this->docs[0].metadataNormalized.empty() 
					&& !this->docs[0].metadataOrg.empty())
				{
					for (auto& doc : this->docs)
					{
						doc.metadataNormalized = doc.metadataOrg;
						for (size_t i = 0; i < degreeByF.size(); ++i)
						{
							doc.metadataOrg[i] = mdIntercepts[i] + doc.metadataOrg[i] * mdCoefs[i];
						}
					}
				}
			}
			
			if (initDocs)
			{
				this->lambda = Eigen::Matrix<Float, -1, -1>::Zero(this->K, this->F);
				this->lambda.col(0).fill(log(this->alpha));
			}
			LBFGSpp::LBFGSParam<Float> param;
			param.max_iterations = this->maxBFGSIteration;
			this->solver = decltype(this->solver){ param };
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, sigma0, degreeByF, mdCoefs, mdIntercepts);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, sigma0, degreeByF, mdCoefs, mdIntercepts, mdMax);

		GDMRModel(size_t _K = 1, const std::vector<uint64_t>& _degreeByF = {}, 
			Float defaultAlpha = 1.0, Float _sigma = 1.0, Float _sigma0 = 1.0, Float _eta = 0.01,
			Float _alphaEps = 1e-10, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, defaultAlpha, _sigma, _eta, _alphaEps, _rg), sigma0(_sigma0), degreeByF(_degreeByF)
		{
			this->F = accumulate(degreeByF.begin(), degreeByF.end(), 1, [](size_t a, size_t b) {return a * (b + 1); });
		}

		GETTER(Fs, const std::vector<uint64_t>&, degreeByF);
		GETTER(Sigma0, Float, sigma0);

		void setSigma0(Float _sigma0) override
		{
			this->sigma0 = _sigma0;
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<std::string>& metadata) const
		{
			if (metadata.size() != degreeByF.size()) 
				throw std::invalid_argument{ "a length of `metadata` should be equal to a length of `degrees`" };

			std::transform(metadata.begin(), metadata.end(), back_inserter(doc.metadataOrg), [](const std::string& w)
			{
				return std::stof(w);
			});
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
			return make_unique<_DocType>(_updateDoc(doc, metadata));
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

		std::vector<Float> getTopicsByDoc(const _DocType& doc) const
		{
			Eigen::Matrix<Float, -1, 1> alphas(this->K);
			thread_local Eigen::Matrix<Float, -1, 1> terms{ this->F };
			getTermsFromMd(&doc.metadataNormalized[0], terms.data());
			for (Tid k = 0; k < this->K; ++k)
			{
				alphas[k] = exp(this->lambda.row(k) * terms) + this->alphaEps;
			}
			std::vector<Float> ret(this->K);
			Float sum = doc.getSumWordWeight() + alphas.sum();
			for (size_t k = 0; k < this->K; ++k)
			{
				ret[k] = (doc.numByTopic[k] + alphas[k]) / sum;
			}
			return ret;
		}

		std::vector<Float> getLambdaByTopic(Tid tid) const override
		{
			std::vector<Float> ret(this->F);
			for (size_t f = 0; f < this->F; ++f) ret[f] = this->lambda.row(tid)[f];
			return ret;
		}

		std::vector<Float> getTDF(const Float* metadata, bool normalize) const override
		{
			Eigen::Matrix<Float, -1, 1> terms{ this->F };
			getTermsFromMd(metadata, terms.data(), true);
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> retMap{ ret.data(), (Eigen::Index)ret.size() };
			retMap = (this->lambda * terms).array();
			if (normalize)
			{
				retMap = (retMap - retMap.maxCoeff()).exp();
				retMap /= retMap.sum();
			}
			return ret;
		}

		std::vector<Float> getTDFBatch(const Float* metadata, size_t stride, size_t cnt, bool normalize) const override
		{
			Eigen::Matrix<Float, -1, -1> terms{ this->F, (Eigen::Index)cnt };
			for (size_t i = 0; i < cnt; ++i)
			{
				getTermsFromMd(metadata + stride * i, terms.col(i).data(), true);
			}
			std::vector<Float> ret(this->K * cnt);
			Eigen::Map<Eigen::Array<Float, -1, -1>> retMap{ ret.data(), (Eigen::Index)this->K, (Eigen::Index)cnt };
			retMap = (this->lambda * terms).array();
			if (normalize)
			{
				retMap.rowwise() -= retMap.colwise().maxCoeff();
				retMap = retMap.exp();
				retMap.rowwise() /= retMap.colwise().sum();
			}
			return ret;
		}
		void setMdRange(const std::vector<Float>& vMin, const std::vector<Float>& vMax) override
		{
			mdIntercepts = vMin;
			mdMax = vMax;
		}

		void getMdRange(std::vector<Float>& vMin, std::vector<Float>& vMax) const override
		{
			vMin = mdIntercepts;
			if (mdMax.empty())
			{
				vMax = mdIntercepts;
				for (size_t i = 0; i < vMax.size(); ++i) vMax[i] += mdCoefs[i];
			}
			else vMax = mdMax;
		}
	};
}
