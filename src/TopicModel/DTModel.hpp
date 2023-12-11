#pragma once
#include "LDAModel.hpp"
#include "DT.h"

/*
Implementation of Dynamic Topic Model using Gibbs sampling by bab2min

* Blei, D. M., & Lafferty, J. D. (2006, June). Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning (pp. 113-120).
* Bhadury, A., Chen, J., Zhu, J., & Liu, S. (2016, April). Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web (pp. 381-390).
* https://github.com/Arnie0426/FastDTM

*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateDTM
	{
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;

		Eigen::Matrix<WeightType, -1, -1> numByTopic; // Dim: (Topic, Time)
		Eigen::Matrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic * Time, Vocabs)
		//ShareableMatrix<WeightType, -1, -1> numByTopicWord; // Dim: (Topic * Time, Vocabs)
		DEFINE_SERIALIZER(numByTopic, numByTopicWord);
	};

	template<TermWeight _tw, typename _RandGen,
		size_t _Flags = flags::partitioned_multisampling,
		typename _Interface = IDTModel,
		typename _Derived = void,
		typename _DocType = DocumentDTM<_tw>,
		typename _ModelState = ModelStateDTM<_tw>>
	class DTModel : public LDAModel<_tw, _RandGen, _Flags, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, DTModel<_tw, _RandGen, _Flags>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, DTModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, _Flags, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr auto tmid()
		{
			return serializer::to_key("DTM\0");
		}

		uint64_t T;
		Float shapeA = 0.03f, shapeB = 0.1f, shapeC = 0.55f;
		Float alphaVar = 1.f, etaVar = 1.f, phiVar = 1.f, etaRegL2 = 0.0f;

		Matrix alphas; // Dim: (Topic, Time)
		Matrix etaByDoc; // Dim: (Topic, Docs) : Topic distribution by docs(and time)
		std::vector<uint32_t> numDocsByTime; // Dim: (Time)
		Matrix phi; // Dim: (Word, Topic * Time)
		std::vector<sample::AliasMethod<>> wordAliasTables; // Dim: (Word * Time)

		template<int _inc>
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, Tid tid) const
		{
			assert(tid < this->K);
			assert(vid < this->realV);
			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<_dec>(doc.numByTopic[tid], _inc * weight);
			updateCnt<_dec>(ld.numByTopic(tid, doc.timepoint), _inc * weight);
			updateCnt<_dec>(ld.numByTopicWord(tid + this->K * doc.timepoint, vid), _inc * weight);
		}

		/*
		Sampling Process

		For each timeslice t,
			For each document d,
				- sampling eta(t, d)
				- sampling zeta
			- sampling phi(t, k) for all k
			- sampling alpha
		*/

		void presampleDocument(_DocType& doc, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt) const
		{
			const Float eps = shapeA * (std::pow(shapeB + 1 + iterationCnt, -shapeC));
			
			// sampling eta
			{
				Vector estimatedCnt = (doc.eta.array() - doc.eta.maxCoeff()).exp();
				Vector etaTmp;
				estimatedCnt *= doc.getSumWordWeight() / estimatedCnt.sum();
				auto prior = (alphas.col(doc.timepoint) - doc.eta) / std::max(etaVar, eps * 2);
				auto grad = doc.numByTopic.template cast<Float>() - estimatedCnt;
				doc.eta.array() += (eps / 2) * (prior.array() + grad.array())
					+ Eigen::Rand::normal<Eigen::Array<Float, -1, 1>>(this->K, 1, rgs) * eps;
			}

			Eigen::Array<Float, -1, 1> expEta = (doc.eta.array() - doc.eta.maxCoeff()).exp();
			doc.aliasTable.buildTable(expEta.data(), expEta.data() + expEta.size());
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			size_t b = 0, e = doc.words.size();
			if (_ps == ParallelScheme::partition)
			{
				b = edd.chunkOffsetByDoc(partitionId, docId);
				e = edd.chunkOffsetByDoc(partitionId + 1, docId);
			}

			size_t vOffset = (_ps == ParallelScheme::partition && partitionId) ? edd.vChunkOffset[partitionId - 1] : 0;

			// sampling zeta
			for (size_t w = b; w < e; ++w)
			{
				auto v = doc.words[w];
				if (v >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, v - vOffset, doc.Zs[w]);

				for (size_t mh = 0; mh < 2; ++mh)
				{
					// doc proposal
					Tid new_z = doc.aliasTable(rgs);
					Float acceptance = std::min(1.f,
						std::exp(phi(v, new_z + this->K * doc.timepoint) - phi(v, doc.Zs[w] + this->K * doc.timepoint))
					);
					if (acceptance >= 1 || rgs.uniform_real() < acceptance) doc.Zs[w] = new_z;

					// word proposal
					new_z = wordAliasTables[v + this->realV * doc.timepoint](rgs);
					acceptance = std::min(1.f,
						std::exp(doc.eta(new_z) - doc.eta(doc.Zs[w]))
					);
					if (acceptance >= 1 || rgs.uniform_real() < acceptance) doc.Zs[w] = new_z;
				}

				addWordTo<1>(ld, doc, w, v - vOffset, doc.Zs[w]);
			}
		}

		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, _RandGen*, const _ExtraDocData& edd) const
		{
			if (_ps == ParallelScheme::copy_merge)
			{
				tState = globalState;
				globalState = localData[0];
				for (size_t i = 1; i < pool.getNumWorkers(); ++i)
				{
					globalState.numByTopicWord += localData[i].numByTopicWord - tState.numByTopicWord;
				}

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
				}
				Eigen::Map<Eigen::Matrix<WeightType, -1, 1>>{ globalState.numByTopic.data(), globalState.numByTopic.size() }
					= globalState.numByTopicWord.rowwise().sum();
			}
			else if (_ps == ParallelScheme::partition)
			{
				std::vector<std::future<void>> res;
				res = pool.enqueueToAll([&](size_t partitionId)
				{
					size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
						e = edd.vChunkOffset[partitionId];
					globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b) = localData[partitionId].numByTopicWord;
				});
				for (auto& r : res) r.get();

				// make all count being positive
				if (_tw != TermWeight::one)
				{
					globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
				}
				Eigen::Map<Eigen::Matrix<WeightType, -1, 1>>{ globalState.numByTopic.data(), globalState.numByTopic.size() }
					= globalState.numByTopicWord.rowwise().sum();
			}
		}

		template<typename _DocIter>
		void _sampleGlobalLevel(ThreadPool* pool, _ModelState*, _RandGen* rgs, _DocIter first, _DocIter last)
		{
			if (!this->realV) return;
			const auto K = this->K;
			const Float eps = shapeA * (std::pow(shapeB + 1 + this->globalStep, -shapeC));

			// sampling phi
			for (size_t k = 0; k < K; ++k)
			{
				Matrix phiGrad{ (Eigen::Index)this->realV, (Eigen::Index)T };
				for (size_t t = 0; t < T; ++t)
				{
					auto phi_tk = phi.col(k + K * t);
					Vector estimatedCnt = (phi_tk.array() - phi_tk.maxCoeff()).exp();
					estimatedCnt *= this->globalState.numByTopic(k, t) / estimatedCnt.sum();

					Vector grad = this->globalState.numByTopicWord.row(k + K * t).template cast<Float>();
					grad -= estimatedCnt;
					auto epsNoise = Eigen::Rand::normal<Eigen::Array<Float, -1, 1>>(this->realV, 1, *rgs) * eps;
					if (t == 0)
					{
						if (T > 1)
						{
							const Float phiVar2 = 100 / (100 + phiVar);
							auto prior = (phi.col(k + K * (t + 1)) * phiVar2 - phi_tk) / std::max(phiVar / 2, eps * 2);
							phiGrad.col(t) = (eps / 2) * (prior.array() + grad.array()) + epsNoise;
						}
						else
						{
							phiGrad.col(t) = (eps / 2) * grad.array() + epsNoise;
						}
					}
					else if (t == T - 1)
					{
						auto prior = (phi.col(k + K * (t - 1)) - phi_tk) / std::max(phiVar, eps * 2);
						phiGrad.col(t) = (eps / 2) * (prior.array() + grad.array()) + epsNoise;
					}
					else
					{
						auto prior = (phi.col(k + K * (t + 1)) + phi.col(k + K * (t - 1)) - 2 * phi_tk) / std::max(phiVar, eps * 2);
						phiGrad.col(t) = (eps / 2) * (prior.array() + grad.array()) + epsNoise;
					}
				}

				for (size_t t = 0; t < T; ++t)
				{
					phi.col(k + K * t) += phiGrad.col(t);
				}
			}

			Matrix newAlphas = Matrix::Zero(alphas.rows(), alphas.cols());
			for (size_t t = 0; t < T; ++t)
			{
				// update alias tables for word proposal
				if (pool && pool->getNumWorkers() > 1)
				{
					const size_t chStride = pool->getNumWorkers() * 8;
					std::vector<std::future<void>> futures;
					futures.reserve(chStride);
					for (size_t ch = 0; ch < chStride; ++ch)
					{
						futures.emplace_back(pool->enqueue([&, ch, chStride](size_t)
						{
							for (Vid v = ch; v < this->realV; v += chStride)
							{
								Eigen::Array<Float, -1, 1> ps = phi.row(v).segment(K * t, K);
								ps = (ps - ps.maxCoeff()).exp();
								wordAliasTables[v + this->realV * t].buildTable(ps.data(), ps.data() + ps.size());
							}
						}));
					}
					for (auto& f : futures) f.get();
				}
				else
				{
					for (Vid v = 0; v < this->realV; ++v)
					{
						Eigen::Array<Float, -1, 1> ps = phi.row(v).segment(K * t, K);
						ps = (ps - ps.maxCoeff()).exp();
						wordAliasTables[v + this->realV * t].buildTable(ps.data(), ps.data() + ps.size());
					}
				}

				// sampling alpha
				Float lambda = 2 / alphaVar + numDocsByTime[t] / etaVar;

				auto newAlpha = newAlphas.col(t);
				newAlpha.setZero();
				for (size_t d = 0; d < this->docs.size(); ++d)
				{
					auto& doc = this->docs[d];
					if (doc.timepoint == t) newAlpha.array() += doc.eta.array();
				}
				newAlpha /= etaVar;
				if(etaRegL2) newAlpha *= 1 - etaRegL2;
				if (t == 0)
				{
					if (T > 1)
					{
						newAlpha += alphas.col(t + 1) / (2 * alphaVar);
					}
					else
					{
						newAlpha.setZero();
					}
				}
				else if (t == T - 1)
				{
					newAlpha += alphas.col(t - 1) / (2 * alphaVar);
				}
				else
				{
					newAlpha += (alphas.col(t + 1) + alphas.col(t - 1)) / alphaVar;
				}
				newAlpha /= lambda;
				newAlpha.array() += Eigen::Rand::normal<Eigen::Array<Float, -1, 1>>(this->K, 1, *rgs) / std::sqrt(lambda);
			}
			alphas = newAlphas;
		}

		template<GlobalSampler _gs, typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, _RandGen* rgs, _DocIter first, _DocIter last) const
		{
			if (_gs != GlobalSampler::inference) return const_cast<DerivedClass*>(this)->_sampleGlobalLevel(pool, localData, rgs, first, last);
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
		}

		template<typename _ExtraDocData>
		void distributePartition(ThreadPool& pool, const _ModelState& globalState, _ModelState* localData, const _ExtraDocData& edd) const
		{
			std::vector<std::future<void>> res = pool.enqueueToAll([&](size_t partitionId)
			{
				size_t b = partitionId ? edd.vChunkOffset[partitionId - 1] : 0,
					e = edd.vChunkOffset[partitionId];

				localData[partitionId].numByTopicWord = globalState.numByTopicWord.block(0, b, globalState.numByTopicWord.rows(), e - b);
				localData[partitionId].numByTopic = globalState.numByTopic;
			});

			for (auto& r : res) r.get();
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);
			if (docId == (size_t)-1)
			{
				doc.eta.init(nullptr, this->K, 1);
			}
			else
			{
				doc.eta.init((Float*)etaByDoc.col(docId).data(), this->K, 1);
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
				// log P(eta | alpha)
				ll -= (doc.eta.matrix() - alphas.col(doc.timepoint)).squaredNorm() / (2 * etaVar);
				ll -= std::log(2 * math::pi * etaVar) / 2 * this->K;

				// log P(z | eta)
				Float etaMax = doc.eta.maxCoeff();
				Eigen::Array<Float, -1, 1> normalizedEta = doc.eta.array()
					- etaMax - std::log((doc.eta.array() - etaMax).exp().sum());
				ll += (doc.numByTopic.template cast<Float>().array() * normalizedEta).sum();
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = 0;
			const size_t V = this->realV;
			if (V == 0) return 0;
			for (Tid t = 0; t < T; ++t)
			{
				// topic-word distribution
				for (Tid k = 0; k < this->K; ++k)
				{
					auto phi_tk = phi.col(k + this->K * t);
					Float phiMax = phi_tk.maxCoeff();
					Eigen::Array<Float, -1, 1> normalizedPhi = phi_tk.array()
						- phiMax - std::log((phi_tk.array() - phiMax).exp().sum());
					ll += (ld.numByTopicWord.row(k + this->K * t).transpose().template cast<Float>().array() * normalizedPhi).sum();

					// log P(phi_t | phi_t-1)
					if (t > 0)
					{
						ll -= (phi_tk - phi.col(k + this->K * (t - 1))).squaredNorm() / (2 * phiVar);
						ll -= std::log(2 * math::pi * phiVar) / 2 * V;
					}
				}

				// log P(alpha_t | alpha_t-1)
				if (t > 0)
				{
					ll -= (alphas.col(t) - alphas.col(t - 1)).squaredNorm() / (2 * alphaVar);
					ll -= std::log(2 * math::pi * alphaVar) / 2 * this->K;
				}
			}
			return ll;
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, T);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K * T, V);

				alphas = Matrix::Zero(this->K, T);
				etaByDoc = Matrix::Zero(this->K, this->docs.size());
				phi = Matrix::Zero(this->realV, this->K * T);
			}

			numDocsByTime.resize(T);
			wordAliasTables.resize(this->realV * this->T);

			size_t docId = 0;
			for (auto& doc : this->docs)
			{
				numDocsByTime[doc.timepoint]++;
				if (!initDocs)
				{
					doc.eta.init((Float*)etaByDoc.col(docId++).data(), this->K, 1);
				}
			}

			for (Tid t = 0; t < T; ++t)
			{
				if (initDocs && !numDocsByTime[t]) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("No document with timepoint = %d", t));

				// update alias tables for word proposal
				for (Vid v = 0; v < this->realV; ++v)
				{
					Eigen::Array<Float, -1, 1> ps = phi.row(v).segment(this->K * t, this->K);
					ps = ps.exp();
					wordAliasTables[v + this->realV * t].buildTable(ps.data(), ps.data() + ps.size());
				}
			}
		}

		template<bool _Infer, typename _Generator>
		void updateStateWithDoc(_Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			z = g.theta(rgs);
			addWordTo<1>(ld, doc, i, w, z);
		}

		std::vector<Float> _getWidsByTopic(size_t tid, bool normalize = true) const
		{
			const size_t V = this->realV;
			std::vector<Float> ret(V);
			Eigen::Map<Eigen::Array<Float, -1, 1>> retMap(ret.data(), V);
			retMap = phi.col(tid).array().exp();
			if (normalize)
			{
				retMap /= retMap.sum();
				Eigen::Array<Float, -1, 1> t = this->globalState.numByTopicWord.row(tid).array().template cast<Float>();
				t /= std::max(t.sum(), (Float)0.1);
				retMap += t;
				retMap /= 2;
			}
			return ret;
		}

		_DocType& _updateDoc(_DocType& doc, uint32_t timepoint) const
		{
			if (timepoint >= T) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "timepoint must < T");
			doc.timepoint = timepoint;
			return doc;
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(this->K * T);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Zs[i] + this->K * doc.timepoint];
				}
			}
			return cnt;
		}

		void updateForCopy()
		{
			BaseClass::updateForCopy();
			size_t docId = 0;
			for (auto& doc : this->docs)
			{
				doc.eta.init((Float*)etaByDoc.col(docId++).data(), this->K, 1);
			}
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, 
			T, shapeA, shapeB, shapeC, alphaVar, etaVar, phiVar, alphas, etaByDoc, phi);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, 
			T, shapeA, shapeB, shapeC, alphaVar, etaVar, phiVar, alphas, etaByDoc, phi);

		GETTER(T, size_t, T);
		GETTER(NumDocsByT, std::vector<uint32_t>, numDocsByTime);
		GETTER(AlphaVar, Float, alphaVar);
		GETTER(EtaVar, Float, etaVar);
		GETTER(PhiVar, Float, phiVar);

		GETTER(ShapeA, Float, shapeA);
		GETTER(ShapeB, Float, shapeB);
		GETTER(ShapeC, Float, shapeC);

		DTModel(const DTArgs& args)
			: BaseClass{ args },
			T{ args.t }, alphaVar{ args.alpha[0] }, etaVar{ args.eta }, phiVar{ args.phi },
			shapeA{ args.shapeA }, shapeB{ args.shapeB }, shapeC{ args.shapeC }, etaRegL2{ args.etaL2Reg }
		{
		}

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) override
		{
			auto doc = this->template _makeFromRawDoc<false>(rawDoc, tokenizer);
			return this->_addDoc(_updateDoc(doc, rawDoc.template getMisc<uint32_t>("timepoint")));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer);
			return std::make_unique<_DocType>(_updateDoc(doc, rawDoc.template getMisc<uint32_t>("timepoint")));
		}

		size_t addDoc(const RawDoc& rawDoc) override
		{
			auto doc = this->_makeFromRawDoc(rawDoc);
			return this->_addDoc(_updateDoc(doc, rawDoc.template getMisc<uint32_t>("timepoint")));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc);
			return std::make_unique<_DocType>(_updateDoc(doc, rawDoc.template getMisc<uint32_t>("timepoint")));
		}

		Float getAlpha(size_t k, size_t t) const override
		{
			if (alphas.size()) return alphas(k, t);
			return 0;
		}

		std::vector<Float> getPhi(size_t k, size_t t) const override
		{
			auto c = phi.col(k + this->K * t);
			return { c.data(), c.data() + c.size() };
		}

		void setShapeA(Float a) override { shapeA = a; }
		void setShapeB(Float b) override { shapeB = b; }
		void setShapeC(Float c) override { shapeC = c; }
	};
}
