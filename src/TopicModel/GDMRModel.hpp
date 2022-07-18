#pragma once
#include "DMRModel.hpp"
#include "../Utils/slp.hpp"
#include "GDMR.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateGDMR : public ModelStateDMR<_tw>
	{
		/*Vector alphas;
		Vector terms;
		std::vector<std::vector<Float>> slpCache;
		std::vector<size_t> ndimCnt;*/
	};

	template<TermWeight _tw, typename _RandGen, 
		size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IGDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentGDMR<_tw>,
		typename _ModelState = ModelStateGDMR<_tw>>
	class GDMRModel : public DMRModel<_tw, _RandGen, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState
	>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, GDMRModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = DMRModel<_tw, _RandGen, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		friend typename BaseClass::BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Float sigma0 = 3, orderDecay = 0;
		std::vector<Float> mdCoefs, mdIntercepts, mdMax;
		std::vector<uint64_t> degreeByF;
		Eigen::Array<Float, -1, 1> orderDecayCached;
		size_t fCont = 1;

		Float getIntegratedLambdaSq(const Eigen::Ref<const Vector, 0, Eigen::InnerStride<>>& lambdas) const
		{
			Float ret = 0;
			for (size_t i = 0; i < this->F; ++i)
			{
				ret += pow(lambdas[this->mdVecSize * i] - log(this->alpha), 2) / 2 / pow(this->sigma0, 2);
				ret += (lambdas.segment(this->mdVecSize * i + 1, fCont - 1).array().pow(2) / 2 * orderDecayCached.segment(1, fCont - 1) / pow(this->sigma, 2)).sum();
				ret += lambdas.segment(this->mdVecSize * i + fCont, this->mdVecSize - fCont).array().pow(2).sum() / 2 / pow(this->sigma, 2);
			}
			return ret;
		}

		void getIntegratedLambdaSqP(const Eigen::Ref<const Vector, 0, Eigen::InnerStride<>>& lambdas, 
			Eigen::Ref<Vector, 0, Eigen::InnerStride<>> ret) const
		{
			for (size_t i = 0; i < this->F; ++i)
			{
				ret[this->mdVecSize * i] = (lambdas[this->mdVecSize * i] - log(this->alpha)) / pow(this->sigma0, 2);
				ret.segment(this->mdVecSize * i + 1, fCont - 1) = lambdas.segment(this->mdVecSize * i + 1, fCont - 1).array() * orderDecayCached.segment(1, fCont - 1) / pow(this->sigma, 2);
				ret.segment(this->mdVecSize * i + fCont, this->mdVecSize - fCont) = lambdas.segment(this->mdVecSize * i + fCont, this->mdVecSize - fCont).array() / pow(this->sigma, 2);
			}
		}

		void initParameters()
		{
			this->lambda = Eigen::Rand::normalLike(this->lambda, this->rg);
			
			for (size_t i = 0; i < this->F; ++i)
			{
				this->lambda.col(this->mdVecSize * i).array() *= sigma0;
				this->lambda.col(this->mdVecSize * i).array() += log(this->alphas.array());

				for (size_t j = 1; j < fCont; ++j)
				{
					this->lambda.col(this->mdVecSize * i + j).array() *= this->sigma / std::sqrt(orderDecayCached[j]);
				}

				for (size_t j = fCont; j < this->mdVecSize; ++j)
				{
					this->lambda.col(this->mdVecSize * i + j).array() *= this->sigma;
				}
			}
		}

		Float getNegativeLambdaLL(Eigen::Ref<Vector> x, Vector& g) const
		{
			auto mappedX = Eigen::Map<Matrix>(x.data(), this->K, this->F * this->fCont);
			auto mappedG = Eigen::Map<Matrix>(g.data(), this->K, this->F * this->fCont);

			Float fx = 0;
			for (size_t k = 0; k < this->K; ++k)
			{
				fx += getIntegratedLambdaSq(mappedX.row(k));
				getIntegratedLambdaSqP(mappedX.row(k), mappedG.row(k));
			}
			return fx;
		}

		/*Float evaluateLambdaObj(Eigen::Ref<Vector> x, Vector& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing search more
			if ((x.array() > this->maxLambda).any()) return INFINITY;

			const auto K = this->K;
			const auto KF = this->K * this->F;

			auto mappedX = Eigen::Map<Matrix>(x.data(), K, this->F);
			Float fx = -static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);

			std::vector<std::future<Vector>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, this](size_t threadId)
				{
					auto& ld = localData[threadId];
					thread_local Vector alphas{ K }, tmpK{ K }, terms{ fCont };
					Vector ret = Vector::Zero(KF + 1);
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						const auto& vx = doc.metadataNormalized;
						size_t xOffset = doc.metadata * fCont;
						getTermsFromMd(&vx[0], terms.data());
						for (Tid k = 0; k < K; ++k)
						{
							alphas[k] = exp(mappedX.row(k).segment(xOffset, fCont) * terms) + this->alphaEps;
							ret[KF] -= math::lgammaT(alphas[k]) - math::lgammaT(doc.numByTopic[k] + alphas[k]);
							assert(std::isfinite(ret[KF]));
							if (!std::isfinite(alphas[k]) && alphas[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphas[k]) - math::digammaT(doc.numByTopic[k] + alphas[k]));
						}
						Float alphaSum = alphas.sum();
						ret[KF] += math::lgammaT(alphaSum) - math::lgammaT(doc.getSumWordWeight() + alphaSum);
						Float t = math::digammaT(alphaSum) - math::digammaT(doc.getSumWordWeight() + alphaSum);
						if (!std::isfinite(alphaSum) && alphaSum > 0)
						{
							ret[KF] = -INFINITY;
							t = 0;
						}
						for (size_t i = 0; i < fCont; ++i)
						{
							ret.segment((i + xOffset) * K, K).array() -= ((tmpK.array() + t) * alphas.array()) * terms[i];
						}
						assert(ret.allFinite());
					}
					return ret;
				}));
			}
			for (auto& r : res)
			{
				auto ret = r.get();
				fx += ret[KF];
				g += ret.head(KF);
			}

			// positive fx is an error from limited precision of float.
			if (fx > 0) return INFINITY;
			return -fx;
		}*/

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

			for (size_t i = 0; i < fCont; ++i)
			{
				out[i] = 1;
				for (size_t n = 0; n < degreeByF.size(); ++n)
				{
					if(digit[n]) out[i] *= slpCache[n][digit[n] - 1];
				}

				for (size_t u = 0; u < digit.size() && ++digit[u] > degreeByF[u]; ++u)
				{
					digit[u] = 0;
				}
			}
		}

		Eigen::Array<Float, -1, 1> calcOrderDecay() const
		{
			Eigen::Array<Float, -1, 1> ret{ fCont };
			std::vector<size_t> digit(degreeByF.size());
			std::fill(digit.begin(), digit.end(), 0);

			for (size_t i = 0; i < fCont; ++i)
			{
				ret[i] = 1;
				for (size_t n = 0; n < degreeByF.size(); ++n)
				{
					ret[i] *= pow(digit[n] + 1, orderDecay * 2);
				}

				for (size_t u = 0; u < digit.size() && ++digit[u] > degreeByF[u]; ++u)
				{
					digit[u] = 0;
				}
			}
			return ret;
		}

		/*template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			thread_local Vector terms{ fCont };
			size_t xOffset = doc.metadata * fCont;
			getTermsFromMd(&doc.metadataNormalized[0], terms.data());
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + (this->lambda.middleCols(xOffset, fCont) * terms).array().exp() + this->alphaEps)
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}*/

		/*template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			double ll = 0;

			Vector alphas(K);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				thread_local Vector terms{ fCont };
				getTermsFromMd(&doc.metadataNormalized[0], terms.data());
				size_t xOffset = doc.metadata * fCont;
				for (Tid k = 0; k < K; ++k)
				{
					alphas[k] = exp(this->lambda.row(k).segment(xOffset, fCont) * terms) + this->alphaEps;
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
		}*/

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
			BaseClass::BaseClass::prepareDoc(doc, docId, wordSize);
			doc.metadataNormalized = normalizeMetadata(doc.metadataOrg);

			doc.mdVec = Vector::Zero(this->mdVecSize);
			getTermsFromMd(doc.metadataNormalized.data(), doc.mdVec.data());
			for (auto x : doc.multiMetadata)
			{
				doc.mdVec[fCont + x] = 1;
			}

			auto p = std::make_pair(doc.metadata, doc.mdVec);
			auto it = this->mdHashMap.find(p);
			if (it == this->mdHashMap.end())
			{
				it = this->mdHashMap.emplace(p, this->mdHashMap.size()).first;
			}
			doc.mdHash = it->second;
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::BaseClass::initGlobalState(initDocs);
			fCont = accumulate(degreeByF.begin(), degreeByF.end(), 1, [](size_t a, size_t b) {return a * (b + 1); });
			if (!this->metadataDict.size())
			{
				this->metadataDict.add("");
			}
			this->F = this->metadataDict.size();
			this->mdVecSize = fCont + this->multiMetadataDict.size();
			if (initDocs)
			{
				collectMinMaxMetadata();
				this->lambda = Matrix::Zero(this->K, this->F * this->mdVecSize);
				for (size_t i = 0; i < this->F; ++i)
				{
					this->lambda.col(this->mdVecSize * i) = log(this->alphas.array());
				}
			}
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

				for (auto& doc : this->docs)
				{
					if (doc.mdVec.size() == this->mdVecSize) continue;
					doc.mdVec = Vector::Zero(this->mdVecSize);
					getTermsFromMd(doc.metadataNormalized.data(), doc.mdVec.data());
					for (auto x : doc.multiMetadata)
					{
						doc.mdVec[fCont + x] = 1;
					}

					auto p = std::make_pair(doc.metadata, doc.mdVec);
					auto it = this->mdHashMap.find(p);
					if (it == this->mdHashMap.end())
					{
						it = this->mdHashMap.emplace(p, this->mdHashMap.size()).first;
					}
					doc.mdHash = it->second;
				}
			}
			
			orderDecayCached = calcOrderDecay();
			LBFGSpp::LBFGSParam<Float> param;
			param.max_iterations = this->maxBFGSIteration;
			this->solver = decltype(this->solver){ param };
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, sigma0, degreeByF, mdCoefs, mdIntercepts);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, sigma0, orderDecay, degreeByF, mdCoefs, mdIntercepts, mdMax);

		GDMRModel(const GDMRArgs& args)
			: BaseClass(args), sigma0(args.sigma0), orderDecay(args.orderDecay), degreeByF(args.degrees)
		{
			fCont = accumulate(degreeByF.begin(), degreeByF.end(), 1, [](size_t a, size_t b) {return a * (b + 1); });
		}

		GETTER(Fs, const std::vector<uint64_t>&, degreeByF);
		GETTER(Sigma0, Float, sigma0);
		GETTER(OrderDecay, Float, orderDecay);

		void setSigma0(Float _sigma0) override
		{
			this->sigma0 = _sigma0;
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<Float>& metadata, const std::string& metadataCat = {}, const std::vector<std::string>& mdVec = {})
		{
			if (metadata.size() != degreeByF.size()) 
				throw exc::InvalidArgument{ "a length of `metadata` should be equal to a length of `degrees`" };
			doc.metadataOrg = metadata;
			
			Vid xid;
			if (_const)
			{
				xid = this->metadataDict.toWid(metadataCat);
				if (xid == non_vocab_id) throw exc::InvalidArgument("unknown metadata '" + metadataCat + "'");

				for (auto& m : mdVec)
				{
					Vid x = this->multiMetadataDict.toWid(m);
					if (x == non_vocab_id) throw exc::InvalidArgument("unknown multi_metadata '" + m + "'");
					doc.multiMetadata.emplace_back(x);
				}
			}
			else
			{
				xid = this->metadataDict.add(metadataCat);

				for (auto& m : mdVec)
				{
					doc.multiMetadata.emplace_back(this->multiMetadataDict.add(m));
				}
			}
			doc.metadata = xid;
			return doc;
		}

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) override
		{
			auto doc = this->template _makeFromRawDoc<false>(rawDoc, tokenizer);
			return this->_addDoc(_updateDoc(doc, 
				rawDoc.template getMisc<std::vector<Float>>("numeric_metadata"),
				rawDoc.template getMiscDefault<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc,
				rawDoc.template getMisc<std::vector<Float>>("numeric_metadata"),
				rawDoc.template getMiscDefault<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		size_t addDoc(const RawDoc& rawDoc) override
		{
			auto doc = this->_makeFromRawDoc(rawDoc);
			return this->_addDoc(_updateDoc(doc, 
				rawDoc.template getMisc<std::vector<Float>>("numeric_metadata"),
				rawDoc.template getMiscDefault<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc,
				rawDoc.template getMisc<std::vector<Float>>("numeric_metadata"),
				rawDoc.template getMiscDefault<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		std::vector<Float> getTDF(const Float* metadata, const std::string& metadataCat, const std::vector<std::string>& multiMetadataCat, bool normalize) const override
		{
			Vector terms = Vector::Zero(this->mdVecSize);
			getTermsFromMd(metadata, terms.data(), true);
			for (auto& s : multiMetadataCat)
			{
				Vid x = this->multiMetadataDict.toWid(s);
				if (x == non_vocab_id) throw exc::InvalidArgument("unknown multi_metadata " + text::quote(s));
				terms[fCont + x] = 1;
			}
			Vid x = this->metadataDict.toWid(metadataCat);
			if (x == non_vocab_id) throw exc::InvalidArgument("unknown metadata " + text::quote(metadataCat));

			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> retMap{ ret.data(), (Eigen::Index)ret.size() };
			retMap = (this->lambda.middleCols(x * this->mdVecSize, this->mdVecSize) * terms).array();
			if (normalize)
			{
				retMap = (retMap - retMap.maxCoeff()).exp();
				retMap /= retMap.sum();
			}
			return ret;
		}

		std::vector<Float> getTDFBatch(const Float* metadata, const std::string& metadataCat, const std::vector<std::string>& multiMetadataCat, size_t stride, size_t cnt, bool normalize) const override
		{
			Matrix terms = Matrix::Zero(this->mdVecSize, (Eigen::Index)cnt);
			for (size_t i = 0; i < cnt; ++i)
			{
				getTermsFromMd(metadata + stride * i, terms.col(i).data(), true);
			}
			for (auto& s : multiMetadataCat)
			{
				Vid x = this->multiMetadataDict.toWid(s);
				if (x == non_vocab_id) throw exc::InvalidArgument("unknown multi_metadata " + text::quote(s));
				terms.row(fCont + x).setOnes();
			}
			Vid x = this->metadataDict.toWid(metadataCat);
			if (x == non_vocab_id) throw exc::InvalidArgument("unknown metadata " + text::quote(metadataCat));

			std::vector<Float> ret(this->K * cnt);
			Eigen::Map<Eigen::Array<Float, -1, -1>> retMap{ ret.data(), (Eigen::Index)this->K, (Eigen::Index)cnt };
			retMap = (this->lambda.middleCols(x * this->mdVecSize, this->mdVecSize) * terms).array();
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
