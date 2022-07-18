#pragma once
#include "LDAModel.hpp"
#include "PT.h"

/*
Implementation of Pseudo-document topic model using Gibbs sampling by bab2min

Zuo, Y., Wu, J., Zhang, H., Lin, H., Wang, F., Xu, K., & Xiong, H. (2016, August). Topic modeling of short texts: A pseudo-document view. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 2105-2114).
*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStatePTM : public ModelStateLDA<_tw>
	{
		using WeightType = typename ModelStateLDA<_tw>::WeightType;

		Eigen::Array<Float, -1, 1> pLikelihood;
		Eigen::ArrayXi numDocsByPDoc;
		Eigen::Matrix<WeightType, -1, -1> numByTopicPDoc;

		//DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_tw>);
	};

	template<TermWeight _tw, typename _RandGen,
		typename _Interface = IPTModel,
		typename _Derived = void,
		typename _DocType = DocumentPT<_tw>,
		typename _ModelState = ModelStatePTM<_tw>>
	class PTModel : public LDAModel<_tw, _RandGen, flags::continuous_doc_data | flags::partitioned_multisampling, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PTModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PTModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, flags::continuous_doc_data | flags::partitioned_multisampling, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr auto tmid()
		{
			return serializer::to_key("PTM");
		}

		uint64_t numPDocs;
		Float lambda;
		uint32_t pseudoDocSamplingInterval = 10;

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			const auto K = this->K;
			for (size_t i = 0; i < 10; ++i)
			{
				Float denom = this->calcDigammaSum(&pool, [&](size_t i) { return this->globalState.numByTopicPDoc.col(i).sum(); }, numPDocs, this->alphas.sum());
				for (size_t k = 0; k < K; ++k)
				{
					Float nom = this->calcDigammaSum(&pool, [&](size_t i) { return this->globalState.numByTopicPDoc(k, i);}, numPDocs, this->alphas(k));
					this->alphas(k) = std::max(nom / denom * this->alphas(k), 1e-5f);
				}
			}
		}

		void samplePseudoDoc(ThreadPool* pool, _ModelState& ld, _RandGen& rgs, _DocType& doc) const
		{
			if (doc.getSumWordWeight() == 0) return;
			Eigen::Array<WeightType, -1, 1> docTopicDist = Eigen::Array<WeightType, -1, 1>::Zero(this->K);
			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				this->template addWordTo<-1>(ld, doc, i, doc.words[i], doc.Zs[i]);
				typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
					= _tw != TermWeight::one ? doc.wordWeights[i] : 1;
				docTopicDist[doc.Zs[i]] += weight;
			}
			--ld.numDocsByPDoc[doc.pseudoDoc];
			
			if (pool && pool->getNumWorkers() > 1)
			{
				std::vector<std::future<void>> futures;
				for (size_t w = 0; w < pool->getNumWorkers(); ++w)
				{
					futures.emplace_back(pool->enqueue([&](size_t, size_t w)
					{
						for (size_t p = w; p < numPDocs; p += pool->getNumWorkers())
						{
							Float ax = math::lgammaSubt(ld.numByTopicPDoc.col(p).array().template cast<Float>() + this->alphas.array(), docTopicDist.template cast<Float>()).sum();
							Float bx = math::lgammaSubt(ld.numByTopicPDoc.col(p).sum() + this->alphas.sum(), docTopicDist.sum());
							ld.pLikelihood[p] = ax - bx;
						}
					}, w));
				}
				for (auto& f : futures) f.get();
			}
			else
			{
				for (size_t p = 0; p < numPDocs; ++p)
				{
					Float ax = math::lgammaSubt(ld.numByTopicPDoc.col(p).array().template cast<Float>() + this->alphas.array(), docTopicDist.template cast<Float>()).sum();
					Float bx = math::lgammaSubt(ld.numByTopicPDoc.col(p).sum() + this->alphas.sum(), docTopicDist.sum());
					ld.pLikelihood[p] = ax - bx;
				}
			}
			ld.pLikelihood = (ld.pLikelihood - ld.pLikelihood.maxCoeff()).exp();
			ld.pLikelihood *= ld.numDocsByPDoc.template cast<Float>() + lambda;

			sample::prefixSum(ld.pLikelihood.data(), numPDocs);
			doc.pseudoDoc = sample::sampleFromDiscreteAcc(ld.pLikelihood.data(), ld.pLikelihood.data() + numPDocs, rgs);

			++ld.numDocsByPDoc[doc.pseudoDoc];
			doc.numByTopic.init(ld.numByTopicPDoc.col(doc.pseudoDoc).data(), this->K, 1);
			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				this->template addWordTo<1>(ld, doc, i, doc.words[i], doc.Zs[i]);
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _DocIter>
		void performSamplingGlobal(ThreadPool* pool, _ModelState& globalState, _RandGen* rgs,
			_DocIter docFirst, _DocIter docLast) const
		{
			if (this->globalStep % pseudoDocSamplingInterval) return;
			for (; docFirst != docLast; ++docFirst)
			{
				samplePseudoDoc(pool, globalState, rgs[0], *docFirst);
			}
		}
		
		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			double ll = 0;
			// doc-topic distribution
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = BaseClass::getLLRest(ld);
			const size_t V = this->realV;
			ll -= math::lgammaT(ld.numDocsByPDoc.sum() + lambda * numPDocs) - math::lgammaT(lambda * numPDocs);
			// pseudo_doc-topic distribution
			for (size_t p = 0; p < numPDocs; ++p)
			{
				ll += math::lgammaT(ld.numDocsByPDoc[p] + lambda) - math::lgammaT(lambda);
				ll -= math::lgammaT(ld.numByTopicPDoc.col(p).sum() + this->alphas.sum()) - math::lgammaT(this->alphas.sum());
				for (Tid k = 0; k < this->K; ++k)
				{
					ll += math::lgammaT(ld.numByTopicPDoc(k, p) + this->alphas[k]) - math::lgammaT(this->alphas[k]);
				}
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			doc.numByTopic.init((WeightType*)this->globalState.numByTopicPDoc.col(0).data(), this->K, 1);
			doc.Zs = tvector<Tid>(wordSize, non_topic_id);
			if (_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			this->globalState.pLikelihood = Vector::Zero(numPDocs);
			this->globalState.numDocsByPDoc = Eigen::ArrayXi::Zero(numPDocs);
			this->globalState.numByTopicPDoc = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, numPDocs);
			BaseClass::initGlobalState(initDocs);
		}

		struct Generator
		{
			std::uniform_int_distribution<uint64_t> psi;
			Eigen::Rand::DiscreteGen<int32_t> theta;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			Generator g;
			g.psi = std::uniform_int_distribution<uint64_t>{ 0, numPDocs - 1 };
			g.theta = Eigen::Rand::DiscreteGen<int32_t>{ this->alphas.data(), this->alphas.data() + this->alphas.size() };
			return g;
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			if (i == 0)
			{
				doc.pseudoDoc = g.psi(rgs);
				++ld.numDocsByPDoc[doc.pseudoDoc];
				doc.numByTopic.init(ld.numByTopicPDoc.col(doc.pseudoDoc).data(), this->K, 1);
			}
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			if (this->etaByTopicWord.size())
			{
				auto col = this->etaByTopicWord.col(w);
				z = sample::sampleFromDiscrete(col.data(), col.data() + col.size(), rgs);
			}
			else
			{
				z = g.theta(rgs);
			}
			this->template addWordTo<1>(ld, doc, i, w, z);
		}

		template<ParallelScheme _ps, bool _infer, typename _DocIter, typename _ExtraDocData>
		void performSampling(ThreadPool& pool, _ModelState* localData, _RandGen* rgs, std::vector<std::future<void>>& res,
			_DocIter docFirst, _DocIter docLast, const _ExtraDocData& edd) const
		{
			// single-threaded sampling
			if (_ps == ParallelScheme::none)
			{
				forShuffled((size_t)std::distance(docFirst, docLast), rgs[0](), [&](size_t id)
				{
					static_cast<const DerivedClass*>(this)->presampleDocument(docFirst[id], id, *localData, *rgs, this->globalStep);
					static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
						docFirst[id], edd, id,
						*localData, *rgs, this->globalStep, 0);

				});
			}
			// multi-threaded sampling on partition and update into global
			else if (_ps == ParallelScheme::partition)
			{
				const size_t chStride = pool.getNumWorkers();
				for (size_t i = 0; i < chStride; ++i)
				{
					res = pool.enqueueToAll([&, i, chStride](size_t partitionId)
					{
						forShuffled((size_t)std::distance(docFirst, docLast), rgs[partitionId](), [&](size_t id)
						{
							if ((docFirst[id].pseudoDoc + partitionId) % chStride != i) return;
							static_cast<const DerivedClass*>(this)->template sampleDocument<_ps, _infer>(
								docFirst[id], edd, id,
								localData[partitionId], rgs[partitionId], this->globalStep, partitionId
							);
						});
					});
					for (auto& r : res) r.get();
					res.clear();
				}
			}
			else
			{
				throw std::runtime_error{ "Unsupported ParallelScheme" };
			}
		}

		void updateForCopy()
		{
			BaseClass::updateForCopy();
			size_t offset = 0;
			for (auto& doc : this->docs)
			{
				doc.template update<>(this->globalState.numByTopicPDoc.col(doc.pseudoDoc).data(), *static_cast<DerivedClass*>(this));
			}
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, numPDocs, lambda);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, numPDocs, lambda);

		GETTER(P, size_t, numPDocs);

		PTModel(const PTArgs& args)
			: BaseClass(args), numPDocs(args.p), lambda(args.lambda)
		{
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (doc.Zs.empty()) return {};
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
			m = this->alphas.array();
			for (size_t i = 0; i < doc.words.size(); ++i)
			{
				if (doc.words[i] >= this->realV) continue;
				typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
					= _tw != TermWeight::one ? doc.wordWeights[i] : 1;
				ret[doc.Zs[i]] += weight;
			}
			if (normalize) m /= m.sum();
			return ret;
		}

		std::vector<Float> getTopicsFromPseudoDoc(const DocumentBase* _doc, bool normalize) const override
		{
			auto& doc = *static_cast<const _DocType*>(_doc);
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
			m = doc.numByTopic.array().template cast<Float>() + this->alphas.array();
			if (normalize)
			{
				m /= m.sum();
			}
			return ret;
		}

		std::vector<std::pair<Tid, Float>> getTopicsFromPseudoDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return extractTopN<Tid>(getTopicsFromPseudoDoc(doc, true), topN);
		}

		void updateDocs()
		{
			for (auto& doc : this->docs)
			{
				doc.template update<>(this->globalState.numByTopicPDoc.col(doc.pseudoDoc).data(), *static_cast<DerivedClass*>(this));
			}
		}
	};
}
