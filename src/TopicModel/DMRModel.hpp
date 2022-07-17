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
		Vector tmpK;
	};
	
	struct MdHash
	{
		size_t operator()(std::pair<uint64_t, Vector> const& p) const
		{
			size_t seed = p.first;
			for (size_t i = 0; i < p.second.size(); ++i) 
			{
				auto elem = p.second[i];
				seed ^= std::hash<decltype(elem)>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

	template<TermWeight _tw, typename _RandGen, 
		size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IDMRModel,
		typename _Derived = void,
		typename _DocType = DocumentDMR<_tw>,
		typename _ModelState = ModelStateDMR<_tw>>
	class DMRModel : public LDAModel<_tw, _RandGen, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_tw, _RandGen, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, DMRModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr auto tmid()
		{
			return serializer::to_key("DMR\0");
		}

		Matrix lambda;
		mutable std::unordered_map<std::pair<uint64_t, Vector>, size_t, MdHash> mdHashMap;
		mutable Matrix cachedAlphas;
		Float sigma;
		uint32_t F = 0, mdVecSize = 1;
		uint32_t optimRepeat = 5;
		Float alphaEps = 1e-10;
		static constexpr Float maxLambda = 10;
		static constexpr size_t maxBFGSIteration = 10;

		Dictionary metadataDict;
		Dictionary multiMetadataDict;
		LBFGSpp::LBFGSSolver<Float, LBFGSpp::LineSearchBracketing> solver;

		Float getNegativeLambdaLL(Eigen::Ref<Vector> x, Vector& g) const
		{
			g = (x.array() - log(this->alpha)) / pow(sigma, 2);
			return (x.array() - log(this->alpha)).pow(2).sum() / 2 / pow(sigma, 2);
		}

		Float evaluateLambdaObj(Eigen::Ref<Vector> x, Vector& g, ThreadPool& pool, _ModelState* localData) const
		{
			// if one of x is greater than maxLambda, return +inf for preventing searching more
			if ((x.array() > maxLambda).any()) return INFINITY;

			const auto K = this->K;

			Float fx = -static_cast<const DerivedClass*>(this)->getNegativeLambdaLL(x, g);
			Eigen::Map<Matrix> xReshaped{ x.data(), (Eigen::Index)K, (Eigen::Index)(F * mdVecSize) };

			std::vector<std::future<Eigen::Array<Float, -1, 1>>> res;
			const size_t chStride = pool.getNumWorkers() * 8;
			for (size_t ch = 0; ch < chStride; ++ch)
			{
				res.emplace_back(pool.enqueue([&, ch](size_t threadId)
				{
					auto& tmpK = localData[threadId].tmpK;
					if (!tmpK.size()) tmpK.resize(this->K);
					Eigen::Array<Float, -1, 1> val = Eigen::Array<Float, -1, 1>::Zero(K * F * mdVecSize + 1);
					Eigen::Map<Matrix> grad{ val.data(), (Eigen::Index)K, (Eigen::Index)(F * mdVecSize) };
					Float& fx = val[K * F * mdVecSize];
					for (size_t docId = ch; docId < this->docs.size(); docId += chStride)
					{
						const auto& doc = this->docs[docId];
						auto alphaDoc = ((xReshaped.middleCols(doc.metadata * mdVecSize, mdVecSize) * doc.mdVec).array().exp() + alphaEps).matrix().eval();
						Float alphaSum = alphaDoc.sum();
						for (Tid k = 0; k < K; ++k)
						{
							fx -= math::lgammaT(alphaDoc[k]) - math::lgammaT(doc.numByTopic[k] + alphaDoc[k]);
							if (!std::isfinite(alphaDoc[k]) && alphaDoc[k] > 0) tmpK[k] = 0;
							else tmpK[k] = -(math::digammaT(alphaDoc[k]) - math::digammaT(doc.numByTopic[k] + alphaDoc[k]));
						}
						fx += math::lgammaT(alphaSum) - math::lgammaT(doc.getSumWordWeight() + alphaSum);
						Float t = math::digammaT(alphaSum) - math::digammaT(doc.getSumWordWeight() + alphaSum);
						if (!std::isfinite(alphaSum) && alphaSum > 0)
						{
							fx = -INFINITY;
							t = 0;
						}
						grad.middleCols(doc.metadata * mdVecSize, mdVecSize) -= (alphaDoc.array() * (tmpK.array() + t)).matrix() * doc.mdVec.transpose();
					}
					return val;
				}));
			}
			for (auto& r : res)
			{
				auto ret = r.get();
				fx += ret[K * F * mdVecSize];
				g += ret.head(K * F * mdVecSize).matrix();
			}

			// positive fx is an error from limited precision of float.
			if (fx > 0) return INFINITY;
			return -fx;
		}

		void initParameters()
		{
			lambda = Eigen::Rand::normalLike(lambda, this->rg, 0, sigma);
			for (size_t f = 0; f < F; ++f)
			{
				lambda.col(f * mdVecSize) += this->alphas.array().log().matrix();
			}
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			Matrix bLambda;
			Float fx = 0, bestFx = INFINITY;
			for (size_t i = 0; i < optimRepeat; ++i)
			{
				static_cast<DerivedClass*>(this)->initParameters();
				int ret = solver.minimize([this, &pool, localData](Eigen::Ref<Vector> x, Vector& g)
				{
					return static_cast<DerivedClass*>(this)->evaluateLambdaObj(x, g, pool, localData);
				}, Eigen::Map<Vector>(lambda.data(), lambda.size()), fx);

				if (fx < bestFx)
				{
					bLambda = lambda;
					bestFx = fx;
					//printf("\t(%d) %e\n", ret, fx);
				}
			}
			if (!std::isfinite(bestFx))
			{
				throw exc::TrainingError{ "optimizing parameters has been failed!" };
			}
			lambda = bLambda;
			updateCachedAlphas();
			//std::cerr << fx << std::endl;
		}

		int restoreFromTrainingError(const exc::TrainingError& e, ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			std::cerr << "Failed to optimize! Reset prior and retry!" << std::endl;
			lambda.setZero();
			updateCachedAlphas();
			return 0;
		}

		auto getCachedAlpha(const _DocType& doc) const
		{
			if (doc.mdHash < cachedAlphas.cols())
			{
				return cachedAlphas.col(doc.mdHash);
			}
			else
			{
				if (!doc.cachedAlpha.size())
				{
					doc.cachedAlpha = (lambda.middleCols(doc.metadata * mdVecSize, mdVecSize) * doc.mdVec).array().exp() + alphaEps;
				}
				return doc.cachedAlpha.col(0);
			}
		}

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto alphas = getCachedAlpha(doc);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}
		
		double getLLDocTopic(const _DocType& doc) const
		{
			const size_t V = this->realV;
			const auto K = this->K;

			auto alphaDoc = getCachedAlpha(doc);
			
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
				auto alphaDoc = getCachedAlpha(doc);
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

		void updateCachedAlphas() const
		{
			cachedAlphas.resize(this->K, mdHashMap.size());

			for (auto& p : mdHashMap)
			{
				cachedAlphas.col(p.second) = (lambda.middleCols(p.first.first * mdVecSize, mdVecSize) * p.first.second).array().exp() + alphaEps;
			}
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);

			doc.mdVec = Vector::Zero(mdVecSize);
			doc.mdVec[0] = 1;
			for (auto x : doc.multiMetadata)
			{
				doc.mdVec[x + 1] = 1;
			}

			auto p = std::make_pair(doc.metadata, doc.mdVec);
			auto it = mdHashMap.find(p);
			if (it == mdHashMap.end())
			{
				it = mdHashMap.emplace(p, mdHashMap.size()).first;
			}
			doc.mdHash = it->second;
		}

		void initGlobalState(bool initDocs)
		{
			BaseClass::initGlobalState(initDocs);
			this->globalState.tmpK = Vector::Zero(this->K);
			F = metadataDict.size();
			mdVecSize = multiMetadataDict.size() + 1;
			if (initDocs)
			{
				lambda.resize(this->K, F * mdVecSize);
				for (size_t f = 0; f < F; ++f)
				{
					lambda.col(f * mdVecSize) = this->alphas.array().log();
					lambda.middleCols(f * mdVecSize + 1, mdVecSize - 1).setZero();
				}
			}
			else
			{
				for (auto& doc : this->docs)
				{
					if (doc.mdVec.size() == mdVecSize) continue;
					doc.mdVec = Vector::Zero(mdVecSize);
					doc.mdVec[0] = 1;
					for (auto x : doc.multiMetadata)
					{
						doc.mdVec[x + 1] = 1;
					}

					auto p = std::make_pair(doc.metadata, doc.mdVec);
					auto it = this->mdHashMap.find(p);
					if (it == this->mdHashMap.end())
					{
						it = this->mdHashMap.emplace(p, mdHashMap.size()).first;
					}
					doc.mdHash = it->second;
				}
			}

			if (_Flags & flags::continuous_doc_data) this->numByTopicDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, this->docs.size());
			LBFGSpp::LBFGSParam<Float> param;
			param.max_iterations = maxBFGSIteration;
			solver = decltype(solver){ param };
		}

		void prepareShared()
		{
			BaseClass::prepareShared();

			for (auto doc : this->docs)
			{
				if (doc.mdHash != (size_t)-1) continue;

				auto p = std::make_pair(doc.metadata, doc.mdVec);
				auto it = mdHashMap.find(p);
				if (it == mdHashMap.end())
				{
					it = mdHashMap.emplace(p, mdHashMap.size()).first;
				}
				doc.mdHash = it->second;
			}

			updateCachedAlphas();
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, sigma, alphaEps, metadataDict, lambda);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, sigma, alphaEps, metadataDict, lambda, multiMetadataDict);

		DMRModel(const DMRArgs& args)
			: BaseClass(args), sigma(args.sigma), alphaEps(args.alphaEps)
		{
			if (sigma <= 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong sigma value (sigma = %f)", sigma));
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::string& metadata, const std::vector<std::string>& mdVec = {})
		{
			Vid xid;
			if (_const)
			{
				xid = metadataDict.toWid(metadata);
				if (xid == (Vid)-1) throw exc::InvalidArgument("unknown metadata '" + metadata + "'");

				for (auto& m : mdVec)
				{
					Vid x = multiMetadataDict.toWid(m);
					if (x == (Vid)-1) throw exc::InvalidArgument("unknown multi_metadata '" + m + "'");
					doc.multiMetadata.emplace_back(x);
				}
			}
			else
			{
				xid = metadataDict.add(metadata);

				for (auto& m : mdVec)
				{
					doc.multiMetadata.emplace_back(multiMetadataDict.add(m));
				}
			}
			doc.metadata = xid;
			return doc;
		}

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) override
		{
			auto doc = this->template _makeFromRawDoc<false>(rawDoc, tokenizer);
			return this->_addDoc(_updateDoc(doc, 
				rawDoc.template getMisc<std::string>("metadata"), 
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc,
				rawDoc.template getMisc<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		size_t addDoc(const RawDoc& rawDoc) override
		{
			auto doc = this->_makeFromRawDoc(rawDoc);
			return this->_addDoc(_updateDoc(doc, 
				rawDoc.template getMisc<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc,
				rawDoc.template getMisc<std::string>("metadata"),
				rawDoc.template getMiscDefault<std::vector<std::string>>("multi_metadata")
			));
		}

		GETTER(F, size_t, F);
		GETTER(MdVecSize, size_t, mdVecSize);
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

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K);
			auto alphaDoc = getCachedAlpha(doc);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
			if (normalize)
			{
				m = (doc.numByTopic.array().template cast<Float>() + alphaDoc.array()) / (doc.getSumWordWeight() + alphaDoc.sum());
			}
			else
			{
				m = doc.numByTopic.array().template cast<Float>() + alphaDoc.array();
			}
			return ret;
		}

		std::vector<Float> getLambdaByMetadata(size_t metadataId) const override
		{
			assert(metadataId < metadataDict.size());
			auto l = lambda.col(metadataId);
			return { l.data(), l.data() + l.size() };
		}

		std::vector<Float> getLambdaByTopic(Tid tid) const override
		{
			std::vector<Float> ret(F * mdVecSize);
			if (this->lambda.size())
			{
				Eigen::Map<Vector>{ ret.data(), (Eigen::Index)ret.size() } = this->lambda.row(tid);
			}
			return ret;
		}

		std::vector<Float> getTopicPrior(const std::string& metadata,
			const std::vector<std::string>& mdVec,
			bool raw = false
		) const override
		{
			Vid xid = metadataDict.toWid(metadata);
			if (xid == (Vid)-1) throw exc::InvalidArgument("unknown metadata '" + metadata + "'");

			Vector xs = Vector::Zero(mdVecSize);
			xs[0] = 1;
			for (auto& m : mdVec)
			{
				Vid x = multiMetadataDict.toWid(m);
				if (x == (Vid)-1) throw exc::InvalidArgument("unknown multi_metadata '" + m + "'");
				xs[x + 1] = 1;
			}

			std::vector<Float> ret(this->K);
			Eigen::Map<Vector> map{ ret.data(), (Eigen::Index)ret.size() };

			if (raw)
			{
				map = lambda.middleCols(xid * mdVecSize, mdVecSize) * xs;
			}
			else
			{
				map = (lambda.middleCols(xid * mdVecSize, mdVecSize) * xs).array().exp() + alphaEps;
			}
			return ret;
		}

		const Dictionary& getMetadataDict() const override { return metadataDict; }
		const Dictionary& getMultiMetadataDict() const override { return multiMetadataDict; }
	};

	/* This is for preventing 'undefined symbol' problem in compiling by clang. */
	template<TermWeight _tw, typename _RandGen, size_t _Flags,
		typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
		constexpr Float DMRModel<_tw, _RandGen, _Flags, _Interface, _Derived, _DocType, _ModelState>::maxLambda;

	template<TermWeight _tw, typename _RandGen, size_t _Flags,
		typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
		constexpr size_t DMRModel<_tw, _RandGen, _Flags, _Interface, _Derived, _DocType, _ModelState>::maxBFGSIteration;

}
