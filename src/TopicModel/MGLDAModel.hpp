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
	template<TermWeight _TW, 
		typename _Interface = IMGLDAModel,
		typename _Derived = void, 
		typename _DocType = DocumentMGLDA<_TW>,
		typename _ModelState = ModelStateLDA<_TW>>
	class MGLDAModel : public LDAModel<_TW, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, MGLDAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, MGLDAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;
		
		FLOAT alphaL;
		FLOAT alphaM, alphaML;
		FLOAT etaL;
		FLOAT gamma;
		TID KL;
		uint32_t T; // window size

		// window and gl./loc. and topic assignment likelihoods for new word. ret T*(K+KL) FLOATs
		FLOAT* getVZLikelihoods(_ModelState& ld, const _DocType& doc, VID vid, uint16_t s) const
		{
			const auto V = this->realV;
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			for (size_t v = 0; v < T; ++v)
			{
				FLOAT pLoc = (doc.numByWinL[s + v] + alphaML) / (doc.numByWin[s + v] + alphaM + alphaML);
				FLOAT pW = doc.numBySentWin(s, v) + gamma;
				if (K)
				{
					zLikelihood.segment(v * (K + KL), K) = (1 - pLoc) * pW
						* (doc.numByTopic.segment(0, K).array().template cast<FLOAT>() + alpha) / (doc.numGl + K * alpha)
						* (ld.numByTopicWord.block(0, vid, K, 1).array().template cast<FLOAT>() + eta) / (ld.numByTopic.segment(0, K).array().template cast<FLOAT>() + V * eta);
				}
				zLikelihood.segment(v * (K + KL) + K, KL) = pLoc * pW
					* (doc.numByWinTopicL.col(s + v).array().template cast<FLOAT>()) / (doc.numByWinL[s + v] + KL * alphaL)
					* (ld.numByTopicWord.block(K, vid, KL, 1).array().template cast<FLOAT>() + etaL) / (ld.numByTopic.segment(K, KL).array().template cast<FLOAT>() + V * etaL);
			}

			sample::prefixSum(zLikelihood.data(), T * (K + KL));
			return &zLikelihood[0];
		}

		template<int INC> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID tid, uint16_t s, uint8_t w, uint8_t r) const
		{
			const auto K = this->K;

			assert(r != 0 || tid < K);
			assert(r == 0 || tid < KL);
			assert(w < T);
			assert(r < 2);
			assert(vid < this->realV);
			assert(s < doc.numBySent.size());

			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByWin[s + w], INC * weight);
			updateCnt<DEC>(doc.numBySentWin(s, w), INC * weight);
			if (r == 0)
			{
				updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
				updateCnt<DEC>(doc.numGl, INC * weight);
				updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
				updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
			}
			else
			{
				updateCnt<DEC>(doc.numByTopic[tid + K], INC * weight);
				updateCnt<DEC>(doc.numByWinL[s + w], INC * weight);
				updateCnt<DEC>(doc.numByWinTopicL(tid, s + w), INC * weight);
				updateCnt<DEC>(ld.numByTopic[tid + K], INC * weight);
				updateCnt<DEC>(ld.numByTopicWord(tid + K, vid), INC * weight);
			}
		}

		void sampleDocument(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
			const auto K = this->K;
			for (size_t w = 0; w < doc.words.size(); ++w)
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
					for (TID k = 0; k < K; ++k)
					{
						ll += math::lgammaT(doc.numByTopic[k] + alpha);
					}
				}

				for (size_t v = 0; v < S + T - 1; ++v)
				{
					ll -= math::lgammaT(doc.numByWinL[v] + KL * alphaL);
					for (TID k = 0; k < KL; ++k)
					{
						ll += math::lgammaT(doc.numByWinTopicL(k, v) + alphaL);
					}
					if (K)
					{
						ll += math::lgammaT(std::max((FLOAT)doc.numByWin[v] - doc.numByWinL[v], (FLOAT)0) + alphaM);
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
			for (TID k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (VID w = 0; w < V; ++w)
				{
					ll += math::lgammaT(ld.numByTopicWord(k, w) + eta);
				}
			}
			ll += (math::lgammaT(V*etaL) - math::lgammaT(etaL)*V) * KL;
			for (TID k = 0; k < KL; ++k)
			{
				ll -= math::lgammaT(ld.numByTopic[k + K] + V * etaL);
				for (VID w = 0; w < V; ++w)
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
				auto&& doc = this->docs[i];
				const size_t S = doc.numBySent.size();
				if (K)
				{
					ll -= math::lgammaT(doc.numGl + K * alpha);
					for (TID k = 0; k < K; ++k)
					{
						ll += math::lgammaT(doc.numByTopic[k] + alpha);
					}
				}

				for (size_t v = 0; v < S + T - 1; ++v)
				{
					ll -= math::lgammaT(doc.numByWinL[v] + KL * alphaL);
					for (TID k = 0; k < KL; ++k)
					{
						ll += math::lgammaT(doc.numByWinTopicL(k, v) + alphaL);
					}
					if (K)
					{
						ll += math::lgammaT(std::max((FLOAT)doc.numByWin[v] - doc.numByWinL[v], (FLOAT)0) + alphaM);
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
			for (TID k = 0; k < K; ++k)
			{
				ll -= math::lgammaT(this->globalState.numByTopic[k] + V * eta);
				for (VID w = 0; w < V; ++w)
				{
					ll += math::lgammaT(this->globalState.numByTopicWord(k, w) + eta);
				}
			}
			ll += (math::lgammaT(V*etaL) - math::lgammaT(etaL)*V) * KL;
			for (TID k = 0; k < KL; ++k)
			{
				ll -= math::lgammaT(this->globalState.numByTopic[k + K] + V * etaL);
				for (VID w = 0; w < V; ++w)
				{
					ll += math::lgammaT(this->globalState.numByTopicWord(k + K, w) + etaL);
				}
			}

			return ll;
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			const size_t S = doc.numBySent.size();
			std::fill(doc.numBySent.begin(), doc.numBySent.end(), 0);
			doc.Zs = tvector<TID>(wordSize);
			doc.Vs.resize(wordSize);
			if (_TW != TermWeight::one) doc.wordWeights.resize(wordSize);
			doc.numByTopic.init(topicDocPtr, this->K + KL);
			doc.numBySentWin = Eigen::Matrix<WeightType, -1, -1>::Zero(S, T);
			doc.numByWin = Eigen::Matrix<WeightType, -1, 1>::Zero(S + T - 1);
			doc.numByWinL = Eigen::Matrix<WeightType, -1, 1>::Zero(S + T - 1);
			doc.numByWinTopicL = Eigen::Matrix<WeightType, -1, -1>::Zero(KL, S + T - 1);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			this->globalState.zLikelihood = Eigen::Matrix<FLOAT, -1, 1>::Zero(T * (this->K + KL));
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K + KL);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K + KL, V);
			}
		}

		struct Generator
		{
			std::discrete_distribution<uint16_t> pi;
			std::uniform_int_distribution<TID> theta;
			std::uniform_int_distribution<TID> thetaL;
			std::uniform_int_distribution<uint16_t> psi;
		};

		Generator makeGeneratorForInit() const
		{
			return Generator{ std::discrete_distribution<uint16_t>{ alphaM, alphaML },
				std::uniform_int_distribution<TID>{ 0, (TID)(this->K - 1) },
				std::uniform_int_distribution<TID>{ 0, (TID)(KL - 1) },
				std::uniform_int_distribution<uint16_t>{ 0, (uint16_t)(T - 1) } };
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			doc.numBySent[doc.sents[i]] += _TW == TermWeight::one ? 1 : doc.wordWeights[i];
			auto& win = doc.Vs[i];
			auto r = g.pi(rgs);
			auto z = (r ? g.thetaL : g.theta)(rgs);
			doc.Zs[i] = z + (r ? this->K : 0);
			win = g.psi(rgs);
			addWordTo<1>(ld, doc, i, doc.words[i], z, doc.sents[i], win, r);
		}


		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, alphaL, alphaM, alphaML, etaL, gamma, KL, T);
	public:
		MGLDAModel(size_t _KG = 1, size_t _KL = 1, size_t _T = 3,
			FLOAT _alphaG = 0.1, FLOAT _alphaL = 0.1, FLOAT _alphaMG = 0.1, FLOAT _alphaML = 0.1,
			FLOAT _etaG = 0.01, FLOAT _etaL = 0.01, FLOAT _gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_KG, _alphaG, _etaG, _rg), KL(_KL), T(_T),
			alphaL(_alphaL), alphaM(_KG ? _alphaMG : 0), alphaML(_alphaML),
			etaL(_etaL), gamma(_gamma)
		{
			if (_KL == 0 || _KL >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong KL value (KL = %zd)", _KL));
			if (_T == 0 || _T >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong T value (T = %zd)", _T));
			if (_alphaL <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong alphaL value (alphaL = %f)", _alphaL));
			if (_etaL <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong etaL value (etaL = %f)", _etaL));
		}

		size_t addDoc(const std::vector<std::string>& words, const std::string& delimiter) override
		{
			_DocType doc;
			size_t numSent = 0;
			for (auto& w : words)
			{
				if (w == delimiter)
				{
					++numSent;
				}
				else
				{
					doc.words.emplace_back(this->dict.add(w));
					doc.sents.emplace_back(numSent);
				}
			}
			doc.numBySent.resize(doc.sents.empty() ? 0 : (doc.sents.back() + 1));
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::string& delimiter) const override
		{
			auto* doc = new _DocType;
			size_t numSent = 0;
			for (auto& w : words)
			{
				if (w == delimiter)
				{
					++numSent;
				}
				else
				{
					auto wid = this->dict.toWid(w);
					if (wid == (VID)-1) continue;
					doc->words.emplace_back(wid);
					doc->sents.emplace_back(numSent);
				}
			}
			doc->numBySent.resize(doc->sents.empty() ? 0 : (doc->sents.back() + 1));
			return std::unique_ptr<DocumentBase>{ doc };
		}

		std::vector<FLOAT> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<FLOAT> ret(this->K + KL);
			Eigen::Map<Eigen::Matrix<FLOAT, -1, 1>> { ret.data(), this->K + KL }.array() =
				doc.numByTopic.array().template cast<FLOAT>() / doc.getSumWordWeight();
			return ret;
		}

		GETTER(KL, size_t, KL);
		GETTER(T, size_t, T);
		GETTER(Gamma, FLOAT, gamma);
		GETTER(AlphaL, FLOAT, alphaL);
		GETTER(EtaL, FLOAT, etaL);
		GETTER(AlphaM, FLOAT, alphaM);
		GETTER(AlphaML, FLOAT, alphaML);
	};

	template<TermWeight _TW>
	template<typename _TopicModel>
	void DocumentMGLDA<_TW>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		this->numByTopic.init(ptr, mdl.getK() + mdl.getKL());
		numBySent.resize(*std::max_element(sents.begin(), sents.end()) + 1);
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			this->numByTopic[this->Zs[i]] += _TW != TermWeight::one ? this->wordWeights[i] : 1;
			numBySent[sents[i]] += _TW != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}