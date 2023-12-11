#pragma once
#include "LDAModel.hpp"
#include "MGLDA.h"
/*
Implementation of MG-LDA using Gibbs sampling by bab2min
Improved version of java implementation(https://github.com/yinfeiy/MG-LDA)

* Titov, I., & McDonald, R. (2008, April). Modeling online reviews with multi-grain topic models. In Proceedings of the 17th international conference on World Wide Web (pp. 111-120). ACM.

*/

namespace tomoto
{
	template<TermWeight _tw, typename _RandGen,
		typename _Interface = IMGLDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentMGLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>>
	class MGLDAModel : public LDAModel<_tw, _RandGen, flags::partitioned_multisampling, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, MGLDAModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, MGLDAModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, flags::partitioned_multisampling, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;
		
		Float alphaL;
		Float alphaM, alphaML;
		Float etaL;
		Float gamma;
		Tid KL;
		uint32_t T; // window size

		// window and gl./loc. and topic assignment likelihoods for new word. ret T*(K+KL) FLOATs
		Float* getVZLikelihoods(_ModelState& ld, const _DocType& doc, Vid vid, uint16_t s) const
		{
			const auto V = this->realV;
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			for (size_t v = 0; v < T; ++v)
			{
				Float pLoc = (doc.numByWinL[s + v] + alphaML) / (doc.numByWin[s + v] + alphaM + alphaML);
				Float pW = doc.numBySentWin(s, v) + gamma;
				if (K)
				{
					zLikelihood.segment(v * (K + KL), K) = (1 - pLoc) * pW
						* (doc.numByTopic.segment(0, K).array().template cast<Float>() + alpha) / (doc.numGl + K * alpha)
						* (ld.numByTopicWord.block(0, vid, K, 1).array().template cast<Float>() + eta) / (ld.numByTopic.segment(0, K).array().template cast<Float>() + V * eta);
				}
				zLikelihood.segment(v * (K + KL) + K, KL) = pLoc * pW
					* (doc.numByWinTopicL.col(s + v).array().template cast<Float>()) / (doc.numByWinL[s + v] + KL * alphaL)
					* (ld.numByTopicWord.block(K, vid, KL, 1).array().template cast<Float>() + etaL) / (ld.numByTopic.segment(K, KL).array().template cast<Float>() + V * etaL);
			}

			sample::prefixSum(zLikelihood.data(), T * (K + KL));
			return &zLikelihood[0];
		}

		template<int _inc> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, Tid tid, uint16_t s, uint8_t w, uint8_t r) const
		{
			const auto K = this->K;

			assert(r != 0 || tid < K);
			assert(r == 0 || tid < KL);
			assert(w < T);
			assert(r < 2);
			assert(vid < this->realV);
			assert(s < doc.numBySent.size());

			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<_dec>(doc.numByWin[s + w], _inc * weight);
			updateCnt<_dec>(doc.numBySentWin(s, w), _inc * weight);
			if (r == 0)
			{
				updateCnt<_dec>(doc.numByTopic[tid], _inc * weight);
				updateCnt<_dec>(doc.numGl, _inc * weight);
				updateCnt<_dec>(ld.numByTopic[tid], _inc * weight);
				updateCnt<_dec>(ld.numByTopicWord(tid, vid), _inc * weight);
			}
			else
			{
				updateCnt<_dec>(doc.numByTopic[tid + K], _inc * weight);
				updateCnt<_dec>(doc.numByWinL[s + w], _inc * weight);
				updateCnt<_dec>(doc.numByWinTopicL(tid, s + w), _inc * weight);
				updateCnt<_dec>(ld.numByTopic[tid + K], _inc * weight);
				updateCnt<_dec>(ld.numByTopicWord(tid + K, vid), _inc * weight);
			}
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

			const auto K = this->K;
			for (size_t w = b; w < e; ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w] - (doc.Zs[w] < K ? 0 : K), doc.sents[w], doc.Vs[w], doc.Zs[w] < K ? 0 : 1);
				auto dist = getVZLikelihoods(ld, doc, doc.words[w], doc.sents[w]);
				auto vz = sample::sampleFromDiscreteAcc(dist, dist + T * (K + KL), rgs);
				doc.Vs[w] = vz / (K + KL);
				doc.Zs[w] = vz % (K + KL);
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w] - (doc.Zs[w] < K ? 0 : K), doc.sents[w], doc.Vs[w], doc.Zs[w] < K ? 0 : 1);
			}
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto K = this->K;
			const auto alpha = this->alpha;
			
			size_t totSents = 0, totWins = 0;
			double ll = 0;
			if (K) ll += (math::lgammaT(K*alpha) - math::lgammaT(alpha)*K) * std::distance(_first, _last);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				const size_t S = doc.numBySent.size();
				if (K)
				{
					ll -= math::lgammaT(doc.numGl + K * alpha);
					for (Tid k = 0; k < K; ++k)
					{
						ll += math::lgammaT(doc.numByTopic[k] + alpha);
					}
				}

				for (size_t v = 0; v < S + T - 1; ++v)
				{
					ll -= math::lgammaT(doc.numByWinL[v] + KL * alphaL);
					for (Tid k = 0; k < KL; ++k)
					{
						ll += math::lgammaT(doc.numByWinTopicL(k, v) + alphaL);
					}
					if (K)
					{
						ll += math::lgammaT(std::max((Float)doc.numByWin[v] - doc.numByWinL[v], (Float)0) + alphaM);
						ll += math::lgammaT(doc.numByWinL[v] + alphaML);
						ll -= math::lgammaT(doc.numByWin[v] + alphaM + alphaML);
					}
				}

				totWins += S + T - 1;
				totSents += S;
				for (size_t s = 0; s < S; ++s)
				{
					ll -= math::lgammaT(doc.numBySent[s] + T * gamma);
					for (size_t v = 0; v < T; ++v)
					{
						ll += math::lgammaT(doc.numBySentWin(s, v) + gamma);
					}
				}
			}
			ll += (math::lgammaT(KL*alphaL) - math::lgammaT(alphaL)*KL) * totWins;
			if (K) ll += (math::lgammaT(alphaM + alphaML) - math::lgammaT(alphaM) - math::lgammaT(alphaML)) * totWins;
			ll += (math::lgammaT(T * gamma) - math::lgammaT(gamma) * T) * totSents;

			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			const auto V = this->realV;
			const auto K = this->K;
			const auto eta = this->eta;
			
			double ll = 0;
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K;
			for (Tid k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (Vid w = 0; w < V; ++w)
				{
					ll += math::lgammaT(ld.numByTopicWord(k, w) + eta);
				}
			}
			ll += (math::lgammaT(V*etaL) - math::lgammaT(etaL)*V) * KL;
			for (Tid k = 0; k < KL; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k + K] + V * etaL);
				for (Vid w = 0; w < V; ++w)
				{
					ll += math::lgammaT(ld.numByTopicWord(k + K, w) + etaL);
				}
			}
			return ll;
		}

		double getLL() const
		{
			double ll = 0;
			const auto V = this->realV;
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			size_t totSents = 0, totWins = 0;
			if(K) ll += (math::lgammaT(K*alpha) - math::lgammaT(alpha)*K) * this->docs.size();
			for (size_t i = 0; i < this->docs.size(); ++i)
			{
				auto& doc = this->docs[i];
				const size_t S = doc.numBySent.size();
				if (K)
				{
					ll -= math::lgammaT(doc.numGl + K * alpha);
					for (Tid k = 0; k < K; ++k)
					{
						ll += math::lgammaT(doc.numByTopic[k] + alpha);
					}
				}

				for (size_t v = 0; v < S + T - 1; ++v)
				{
					ll -= math::lgammaT(doc.numByWinL[v] + KL * alphaL);
					for (Tid k = 0; k < KL; ++k)
					{
						ll += math::lgammaT(doc.numByWinTopicL(k, v) + alphaL);
					}
					if (K)
					{
						ll += math::lgammaT(std::max((Float)doc.numByWin[v] - doc.numByWinL[v], (Float)0) + alphaM);
						ll += math::lgammaT(doc.numByWinL[v] + alphaML);
						ll -= math::lgammaT(doc.numByWin[v] + alphaM + alphaML);
					}
				}

				totWins += S + T - 1;
				totSents += S;
				for (size_t s = 0; s < S; ++s)
				{
					ll -= math::lgammaT(doc.numBySent[s] + T * gamma);
					for (size_t v = 0; v < T; ++v)
					{
						ll += math::lgammaT(doc.numBySentWin(s, v) + gamma);
					}
				}
			}
			ll += (math::lgammaT(KL*alphaL) - math::lgammaT(alphaL)*KL) * totWins;
			if(K) ll += (math::lgammaT(alphaM + alphaML) - math::lgammaT(alphaM) - math::lgammaT(alphaML)) * totWins;
			ll += (math::lgammaT(T * gamma) - math::lgammaT(gamma) * T) * totSents;

			//
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K;
			for (Tid k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(this->globalState.numByTopic[k] + V * eta);
				for (Vid w = 0; w < V; ++w)
				{
					ll += math::lgammaT(this->globalState.numByTopicWord(k, w) + eta);
				}
			}
			ll += (math::lgammaT(V*etaL) - math::lgammaT(etaL)*V) * KL;
			for (Tid k = 0; k < KL; ++k)
			{
				ll -= math::lgammaT(this->globalState.numByTopic[k + K] + V * etaL);
				for (Vid w = 0; w < V; ++w)
				{
					ll += math::lgammaT(this->globalState.numByTopicWord(k + K, w) + etaL);
				}
			}

			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			auto tmp = doc.sents;
			for (size_t i = 0; i < doc.wOrder.size(); ++i)
			{
				doc.sents[doc.wOrder[i]] = tmp[i];
			}

			const size_t S = doc.numBySent.size();
			std::fill(doc.numBySent.begin(), doc.numBySent.end(), 0);
			doc.Zs = tvector<Tid>(wordSize, non_topic_id);
			doc.Vs.resize(wordSize);
			if (_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
			doc.numByTopic.init(nullptr, this->K + KL, 1);
			doc.numBySentWin = Eigen::Matrix<WeightType, -1, -1>::Zero(S, T);
			doc.numByWin = Eigen::Matrix<WeightType, -1, 1>::Zero(S + T - 1);
			doc.numByWinL = Eigen::Matrix<WeightType, -1, 1>::Zero(S + T - 1);
			doc.numByWinTopicL = Eigen::Matrix<WeightType, -1, -1>::Zero(KL, S + T - 1);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Vector::Zero(T * (this->K + KL));
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K + KL);
				//this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K + KL, V);
				this->globalState.numByTopicWord.init(nullptr, this->K + KL, V);
			}
		}

		struct Generator
		{
			std::discrete_distribution<uint16_t> pi;
			std::uniform_int_distribution<Tid> theta;
			std::uniform_int_distribution<Tid> thetaL;
			std::uniform_int_distribution<uint16_t> psi;
		};

		Generator makeGeneratorForInit(const _DocType*) const
		{
			return Generator{ std::discrete_distribution<uint16_t>{ alphaM, alphaML },
				std::uniform_int_distribution<Tid>{ 0, (Tid)(this->K - 1) },
				std::uniform_int_distribution<Tid>{ 0, (Tid)(KL - 1) },
				std::uniform_int_distribution<uint16_t>{ 0, (uint16_t)(T - 1) } };
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			doc.numBySent[doc.sents[i]] += _tw == TermWeight::one ? 1 : doc.wordWeights[i];
			auto w = doc.words[i];
			size_t r, z;
			if (this->etaByTopicWord.size())
			{
				Eigen::Array<Float, -1, 1> col = this->etaByTopicWord.col(w);
				col.head(this->K) *= alphaM / this->K;
				col.tail(this->KL) *= alphaML / this->KL;
				doc.Zs[i] = z = sample::sampleFromDiscrete(col.data(), col.data() + col.size(), rgs);
				r = z < this->K;
				if (z >= this->K) z -= this->K;
			}
			else
			{
				r = g.pi(rgs);
				z = (r ? g.thetaL : g.theta)(rgs);
				doc.Zs[i] = z + (r ? this->K : 0);
			}
			
			auto& win = doc.Vs[i];
			win = g.psi(rgs);
			addWordTo<1>(ld, doc, i, w, z, doc.sents[i], win, r);
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(this->K + KL);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.Zs[i]];
				}
			}
			return cnt;
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, alphaL, alphaM, alphaML, etaL, gamma, KL, T);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, alphaL, alphaM, alphaML, etaL, gamma, KL, T);

		MGLDAModel(const MGLDAArgs& args)
			: BaseClass(args), KL(args.kL), T(args.t),
			alphaL(args.alphaL[0]), alphaM(args.k ? args.alphaMG : 0), alphaML(args.alphaML),
			etaL(args.etaL), gamma(args.gamma)
		{
			if (KL == 0 || KL >= 0x80000000) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong KL value (KL = %zd)", KL));
			if (T == 0 || T >= 0x80000000) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong T value (T = %zd)", T));

			if (args.alpha.size() != 1)
			{
				THROW_ERROR_WITH_INFO(exc::Unimplemented, "An asymmetric alpha prior is not supported yet at MGLDA.");
			}

			if (args.alphaL.size() == 1)
			{
			}
			else if (args.alphaL.size() == args.kL)
			{
				THROW_ERROR_WITH_INFO(exc::Unimplemented, "An asymmetric alphaL prior is not supported yet at MGLDA.");
			}
			else
			{
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong alphaL value (len = %zd)", args.alphaL.size()));
			}

			if (alphaL <= 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong alphaL value (alphaL = %f)", alphaL));
			if (etaL <= 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong etaL value (etaL = %f)", etaL));
		}

		template<bool _const, typename _FnTokenizer>
		_DocType _makeFromRawDoc(const RawDoc& rawDoc, _FnTokenizer&& tokenizer, const std::string& delimiter)
		{
			_DocType doc;
			size_t numSent = 0;
			doc.rawStr = rawDoc.rawStr;
			for (auto& p : tokenizer(doc.rawStr))
			{
				if (std::get<0>(p) == delimiter)
				{
					++numSent;
					continue;
				}

				Vid wid;
				if (_const)
				{
					wid = this->dict.toWid(std::get<0>(p));
					if (wid == (Vid)-1) continue;
				}
				else
				{
					wid = this->dict.add(std::get<0>(p));
				}
				auto pos = std::get<1>(p);
				auto len = std::get<2>(p);
				doc.words.emplace_back(wid);
				doc.sents.emplace_back(numSent);
				doc.origWordPos.emplace_back(pos);
				doc.origWordLen.emplace_back(len);
			}
			doc.numBySent.resize(doc.sents.empty() ? 0 : (doc.sents.back() + 1));
			return doc;
		}

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer)
		{
			return this->_addDoc(_makeFromRawDoc<false>(rawDoc, tokenizer, rawDoc.template getMisc<std::string>("delimiter")));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const
		{
			return std::make_unique<_DocType>(as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer, rawDoc.template getMisc<std::string>("delimiter")));
		}

		template<bool _const = false>
		_DocType _makeFromRawDoc(const RawDoc& rawDoc)
		{
			_DocType doc;
			doc.rawStr = rawDoc.rawStr;
			auto delimiter = rawDoc.template getMisc<std::string>("delimiter");
			size_t numSent = 0;
			Vid delimiterId = this->dict.toWid(delimiter);
			if (!rawDoc.rawWords.empty())
			{
				for (size_t i = 0; i < rawDoc.rawWords.size(); ++i)
				{
					auto& w = rawDoc.rawWords[i];
					if (w == delimiter)
					{
						++numSent;
						continue;
					}

					Vid id;
					if (_const)
					{
						id = this->dict.toWid(w);
						if (id == (Vid)-1) continue;
					}
					else
					{
						id = this->dict.add(w);
					}
					doc.words.emplace_back(id);
					doc.sents.emplace_back(numSent);
					if (rawDoc.rawWords.size() == rawDoc.origWordPos.size())
					{
						doc.origWordPos.emplace_back(rawDoc.origWordPos[i]);
						doc.origWordLen.emplace_back(rawDoc.origWordLen[i]);
					}
				}
			}
			else if (!rawDoc.words.empty())
			{
				for (size_t i = 0; i < rawDoc.words.size(); ++i)
				{
					auto& w = rawDoc.words[i];
					if (w == delimiterId)
					{
						++numSent;
						continue;
					}
					doc.words.emplace_back(w);
					doc.sents.emplace_back(numSent);
					if (rawDoc.words.size() == rawDoc.origWordPos.size())
					{
						doc.origWordPos.emplace_back(rawDoc.origWordPos[i]);
						doc.origWordLen.emplace_back(rawDoc.origWordLen[i]);
					}
				}
			}
			doc.numBySent.resize(doc.sents.empty() ? 0 : (doc.sents.back() + 1));
			return doc;
		}

		size_t addDoc(const RawDoc& rawDoc)
		{
			return this->_addDoc(_makeFromRawDoc(rawDoc));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const
		{
			return std::make_unique<_DocType>(as_mutable(this)->template _makeFromRawDoc<true>(rawDoc));
		}

		size_t getNumTopicsForPrior() const override
		{
			return this->K + KL;
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			if (priors.size() != this->K + KL) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors.size() must be equal to K + KL.");
			for (auto p : priors)
			{
				if (p < 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "priors must not be less than 0.");
			}
			this->dict.add(word);
			this->etaByWord.emplace(word, priors);
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K + KL);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K + KL };
			if (normalize)
			{
				m = doc.numByTopic.array().template cast<Float>() / doc.getSumWordWeight();
			}
			else
			{
				m = doc.numByTopic.array().template cast<Float>();
			}
			return ret;
		}

		GETTER(KL, size_t, KL);
		GETTER(T, size_t, T);
		GETTER(Gamma, Float, gamma);
		GETTER(AlphaL, Float, alphaL);
		GETTER(EtaL, Float, etaL);
		GETTER(AlphaM, Float, alphaM);
		GETTER(AlphaML, Float, alphaML);
	};

	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentMGLDA<_tw>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		this->numByTopic.init(ptr, mdl.getK() + mdl.getKL(), 1);
		numBySent.resize(*std::max_element(sents.begin(), sents.end()) + 1);
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			this->numByTopic[this->Zs[i]] += _tw != TermWeight::one ? this->wordWeights[i] : 1;
			numBySent[sents[i]] += _tw != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}
