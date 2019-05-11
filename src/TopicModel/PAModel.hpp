#pragma once
#include "LDAModel.hpp"

/*
Implementation of Pachinko Allocation using Gibbs sampling by bab2min

Li, W., & McCallum, A. (2006, June). Pachinko allocation: DAG-structured mixture models of topic correlations. In Proceedings of the 23rd international conference on Machine learning (pp. 577-584). ACM.
*/

namespace tomoto
{
	template<TermWeight _TW>
	struct DocumentPA : public DocumentLDA<_TW>
	{
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;

		tvector<TID> Z2s;
		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;
	};

	template<TermWeight _TW>
	struct ModelStatePA : public ModelStateLDA<_TW>
	{
		using WeightType = typename ModelStateLDA<_TW>::WeightType;
		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;
		Eigen::Matrix<WeightType, -1, 1> numByTopic2;
		Eigen::Matrix<FLOAT, -1, 1> subTmp;
	};

	class IPAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentPA<TermWeight::one>;
		static IPAModel* create(TermWeight _weight, size_t _K1 = 1, size_t _K2 = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual size_t getDirichletEstIteration() const = 0;
		virtual void setDirichletEstIteration(size_t iter) = 0;
		virtual size_t getK2() const = 0;
		virtual FLOAT getSubAlpha(TID k1, TID k2) const = 0;
		virtual std::vector<FLOAT> getSubTopicBySuperTopic(TID k) const = 0;
	};

	template<TermWeight _TW,
		typename _Interface = IPAModel,
		typename _Derived = void,
		typename _DocType = DocumentPA<_TW>,
		typename _ModelState = ModelStatePA<_TW>>
	class PAModel : public LDAModel<_TW, false, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, false, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		size_t K2;
		FLOAT epsilon = 0.00001;
		size_t iteration = 5;

		Eigen::Matrix<FLOAT, -1, 1> subAlphaSum; // len = K
		Eigen::Matrix<FLOAT, -1, -1> subAlphas; // len = K * K2
		void optimizeHyperparameter(ThreadPool& pool, _ModelState* localData)
		{
			const auto K = this->K;
			auto calcDigammaSum = [](auto list, size_t len, FLOAT alpha)
			{
				FLOAT ret = 0;
				auto dAlpha = math::digammaT(alpha);
				for (size_t i = 0; i < len; ++i)
				{
					ret += math::digammaT(list(i) + alpha) - dAlpha;
				}
				return ret;
			};

			std::vector<std::future<void>> res;
			for (size_t k = 0; k < K; ++k)
			{
				pool.enqueue([&, k](size_t)
				{
					for (size_t i = 0; i < iteration; ++i)
					{
						FLOAT denom = calcDigammaSum([&](size_t i) { return this->docs[i].numByTopic[k]; }, this->docs.size(), subAlphaSum[k]);
						for (size_t k2 = 0; k2 < K2; ++k2)
						{
							FLOAT nom = calcDigammaSum([&](size_t i) { return this->docs[i].numByTopic1_2(k, k2); }, this->docs.size(), subAlphas(k, k2));
							subAlphas(k, k2) = std::max(nom / denom * subAlphas(k, k2), epsilon);
						}
						subAlphaSum[k] = subAlphas.row(k).sum();
					}
				});
			}
			for (auto& r : res) r.get();
		}

		// topic 1 & 2 assignment likelihoods for new word. ret K*K2 FLOATs
		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t vid) const
		{
			const size_t V = this->dict.size();
			const auto eta = this->eta;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			
			ld.subTmp = (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + eta) / (ld.numByTopic2.array().template cast<FLOAT>() + V * eta);

			for (size_t k = 0; k < this->K; ++k)
			{
				zLikelihood.segment(K2 * k, K2) = (doc.numByTopic[k] + this->alpha)
					* (doc.numByTopic1_2.row(k).transpose().array().template cast<FLOAT>() + subAlphas.row(k).transpose().array()) / (doc.numByTopic[k] + subAlphaSum[k])
					* ld.subTmp.array();
			}
			sample::prefixSum(zLikelihood.data(), zLikelihood.size());
			return &zLikelihood[0];
		}

		template<int INC> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID z1, TID z2) const
		{
			size_t V = this->dict.size();
			assert(vid < V);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[z1], INC * weight);
			updateCnt<DEC>(doc.numByTopic1_2(z1, z2), INC * weight);
			updateCnt<DEC>(ld.numByTopic[z1], INC * weight);
			updateCnt<DEC>(ld.numByTopic2[z2], INC * weight);
			updateCnt<DEC>(ld.numByTopic1_2(z1, z2), INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(z2, vid), INC * weight);
		}

		void sampleDocument(_DocType& doc, _ModelState& ld, RANDGEN& rgs) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.Z2s[w]);
				auto dist = getZLikelihoods(ld, doc, doc.words[w]);
				auto z = sample::sampleFromDiscreteAcc(dist, dist + this->K * K2, rgs);
				doc.Zs[w] = z / K2;
				doc.Z2s[w] = z % K2;
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.Z2s[w]);
			}
		}

		void updateGlobal(ThreadPool& pool, _ModelState* localData)
		{
			std::vector<std::future<void>> res(pool.getNumWorkers());

			this->tState = this->globalState;
			this->globalState = localData[0];
			for (size_t i = 1; i < pool.getNumWorkers(); ++i)
			{
				this->globalState.numByTopic += localData[i].numByTopic - this->tState.numByTopic;
				this->globalState.numByTopic1_2 += localData[i].numByTopic1_2 - this->tState.numByTopic1_2;
				this->globalState.numByTopic2 += localData[i].numByTopic2 - this->tState.numByTopic2;
				this->globalState.numByTopicWord += localData[i].numByTopicWord - this->tState.numByTopicWord;
			}

			// make all count being positive
			if (_TW != TermWeight::one)
			{
				this->globalState.numByTopic = this->globalState.numByTopic.cwiseMax(0);
				this->globalState.numByTopic1_2 = this->globalState.numByTopic1_2.cwiseMax(0);
				this->globalState.numByTopic2 = this->globalState.numByTopic2.cwiseMax(0);
				this->globalState.numByTopicWord = this->globalState.numByTopicWord.cwiseMax(0);
			}

			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, this, i](size_t threadId)
				{
					localData[i] = this->globalState;
				});
			}
			for (auto&& r : res) r.get();
		}

		double getLL() const
		{
			double ll = 0;
			const size_t V = this->dict.size();
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			ll += (math::lgammaT(K*alpha) - math::lgammaT(alpha)*K) * this->docs.size();
			for (auto& doc : this->docs)
			{
				ll -= math::lgammaT(doc.template getSumWordWeight<_TW>() + K * alpha);
				for (TID k = 0; k < K; ++k)
				{
					ll += math::lgammaT(doc.numByTopic[k] + alpha);
				}
			}
			for (TID k = 0; k < K; ++k)
			{
				ll += math::lgammaT(subAlphaSum[k]);
				ll -= math::lgammaT(this->globalState.numByTopic[k] + subAlphaSum[k]);
				for (TID k2 = 0; k2 < K2; ++k2)
				{
					ll -= math::lgammaT(subAlphas(k, k2));
					ll += math::lgammaT(this->globalState.numByTopic1_2(k, k2) + subAlphas(k, k2));
				}
			}
			ll += (math::lgammaT(V*eta) - math::lgammaT(eta)*V) * K2;
			for (TID k2 = 0; k2 < K2; ++k2)
			{
				ll -= math::lgammaT(this->globalState.numByTopic2[k2] + V * eta);
				for (VID v = 0; v < V; ++v)
				{
					ll += math::lgammaT(this->globalState.numByTopicWord(k2, v) + eta);
				}
			}
			return ll;
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);

			doc.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
			doc.Z2s = tvector<TID>(wordSize);
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->dict.size();
			this->globalState.zLikelihood = Eigen::Matrix<FLOAT, -1, 1>::Zero(this->K * K2);
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
				this->globalState.numByTopic2 = Eigen::Matrix<WeightType, -1, 1>::Zero(K2);
				this->globalState.numByTopic1_2 = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, K2);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K2, V);
			}
		}

		struct Generator
		{
			std::uniform_int_distribution<TID> theta, theta2;
		};

		Generator makeGeneratorForInit() const
		{
			return Generator{ 
				std::uniform_int_distribution<TID>{0, (TID)(this->K - 1)},
				std::uniform_int_distribution<TID>{0, (TID)(K2 - 1)},
			};
		}

		void updateStateWithDoc(Generator& g, _ModelState& ld, RANDGEN& rgs, _DocType& doc, size_t i) const
		{
			auto w = doc.words[i];
			doc.Zs[i] = g.theta(rgs);
			doc.Z2s[i] = g.theta2(rgs);
			addWordTo<1>(ld, doc, i, w, doc.Zs[i], doc.Z2s[i]);
		}

	public:
		PAModel(size_t _K1 = 1, size_t _K2 = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RANDGEN& _rg = RANDGEN{ std::random_device{}() })
			: BaseClass(_K1, _alpha, _eta, _rg), K2(_K2)
		{
			subAlphaSum = Eigen::Matrix<FLOAT, -1, 1>::Constant(_K1, _K2 * 0.1);
			subAlphas = Eigen::Matrix<FLOAT, -1, -1>::Constant(_K1, _K2, 0.1);
			this->optimInterval = 1;
		}

		GETTER(K2, size_t, K2);
		GETTER(DirichletEstIteration, size_t, iteration);

		void setDirichletEstIteration(size_t iter) override
		{
			if (!iter) throw std::invalid_argument("iter must > 0");
			iteration = iter;
		}

		FLOAT getSubAlpha(TID k1, TID k2) const override { return subAlphas(k1, k2); }

		std::vector<FLOAT> getSubTopicBySuperTopic(TID k) const
		{
			assert(k < this->K);
			FLOAT sum = this->globalState.numByTopic[k] + subAlphaSum[k];
			Eigen::Matrix<FLOAT, -1, 1> ret = (this->globalState.numByTopic1_2.row(k).array().template cast<FLOAT>() + subAlphas.row(k).array()) / sum;
			return { ret.data(), ret.data() + K2 };
		}

		std::vector<FLOAT> _getWidsByTopic(TID k2) const
		{
			assert(k2 < K2);
			const size_t V = this->dict.size();
			std::vector<FLOAT> ret(V);
			FLOAT sum = this->globalState.numByTopic2[k2] + V * this->eta;
			auto r = this->globalState.numByTopicWord.row(k2);
			for (size_t v = 0; v < V; ++v)
			{
				ret[v] = (r[v] + this->eta) / sum;
			}
			return ret;
		}
	};
	
	IPAModel* IPAModel::create(TermWeight _weight, size_t _K, size_t _K2, FLOAT _alpha, FLOAT _eta, const RANDGEN& _rg)
	{
		SWITCH_TW(_weight, PAModel, _K, _K2, _alpha, _eta, _rg);
	}
}