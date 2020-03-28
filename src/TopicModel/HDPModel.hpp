#pragma once
#include "LDAModel.hpp"
#include "HDP.h"

/*
Implementation of HDP using Gibbs sampling by bab2min

* Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
* Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.
*/

namespace tomoto
{
	template<TermWeight _tw>
	struct ModelStateHDP : public ModelStateLDA<_tw>
	{
		Eigen::Matrix<Float, -1, 1> tableLikelihood, topicLikelihood;
		Eigen::Matrix<int32_t, -1, 1> numTableByTopic;
		size_t totalTable = 0;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_tw>, numTableByTopic, totalTable);
	};

	template<TermWeight _tw,
		typename _Interface = IHDPModel,
		typename _Derived = void,
		typename _DocType = DocumentHDP<_tw>,
		typename _ModelState = ModelStateHDP<_tw>>
	class HDPModel : public LDAModel<_tw, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_tw>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Float gamma;

		template<typename _NumFunc>
		static Float estimateConcentrationParameter(_NumFunc ns, Float tableCnt, size_t size, Float alpha, RandGen& rgs)
		{
			Float a = 1, b = 1;
			for (size_t i = 0; i < 10; ++i)
			{
				Float sumLogW = 0;
				Float sumS = 0;
				for (size_t j = 0; j < size; ++j)
				{
					Float w = math::beta_distribution<Float>{ alpha + 1, (Float)ns(j) }(rgs);
					Float s = std::bernoulli_distribution{ ns(j) / (ns(j) + alpha) }(rgs) ? 1 : 0;
					sumLogW += log(w);
					sumS += s;
				}
				a += tableCnt - sumS;
				b -= sumLogW;
				alpha = std::gamma_distribution<Float>{ a, 1 / b }(rgs);
			}
			return alpha;
		}

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			size_t tableCnt = 0;
			for (auto& doc : this->docs)
			{
				tableCnt += doc.getNumTable();
			}

			this->alpha = estimateConcentrationParameter([this](size_t s)
			{
				return this->docs[s].getSumWordWeight();
			}, tableCnt, this->docs.size(), this->alpha, *rgs);

			gamma = estimateConcentrationParameter([this](size_t)
			{
				return this->globalState.totalTable;
			}, this->getLiveK(), 1, gamma, *rgs);
		}

		size_t addTopic(_ModelState& ld) const
		{
			const size_t V = this->realV;
			size_t pos;
			for (pos = 0; pos < ld.numTableByTopic.size(); ++pos)
			{
				if (!ld.numTableByTopic[pos]) break;
			}
			
			if (pos >= ld.numByTopic.size())
			{
				size_t oldSize = ld.numByTopic.size(), newSize = pos + 1;
				ld.numTableByTopic.conservativeResize(newSize);
				ld.numTableByTopic.tail(newSize - oldSize).setZero();
				ld.numByTopic.conservativeResize(newSize);
				ld.numByTopic.tail(newSize - oldSize).setZero();
				ld.numByTopicWord.conservativeResize(newSize, Eigen::NoChange);
				ld.numByTopicWord.block(oldSize, 0, newSize - oldSize, V).setZero();
			}
			else
			{
				ld.numByTopic[pos] = 0;
				ld.numByTopicWord.row(pos).setZero();
			}
			return pos;
		}

		void calcWordTopicProb(_ModelState& ld, Vid vid) const
		{
			const size_t V = this->realV;
			const auto K = ld.numByTopic.size();
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood.resize(K + 1);
			zLikelihood.head(K) = (ld.numByTopicWord.col(vid).array().template cast<Float>() + this->eta)
				/ (ld.numByTopic.array().template cast<Float>() + V * this->eta);
			zLikelihood[K] = 1. / V;
		}

		Float* getTableLikelihoods(_ModelState& ld, _DocType& doc, Vid vid) const
		{
			assert(vid < this->realV);
			const size_t T = doc.numTopicByTable.size();
			const auto K = ld.numByTopic.size();
			Float acc = 0;
			ld.tableLikelihood.resize(T + 1);
			for (size_t t = 0; t < T; ++t)
			{
				ld.tableLikelihood[t] = acc += doc.numTopicByTable[t].num * ld.zLikelihood[doc.numTopicByTable[t].topic];
			}
			Float pNewTable = ld.zLikelihood[K] / (gamma + ld.totalTable);
			ld.tableLikelihood[T] = acc += this->alpha * pNewTable;
			return &ld.tableLikelihood[0];
		}

		Float* getTopicLikelihoods(_ModelState& ld) const
		{
			const size_t V = this->realV;
			const auto K = ld.numByTopic.size();
			ld.topicLikelihood.resize(K + 1);
			ld.topicLikelihood.head(K) = ld.zLikelihood.head(K).array().template cast<Float>() * ld.numTableByTopic.array().template cast<Float>();
			ld.topicLikelihood[K] = ld.zLikelihood[K] * gamma;
			sample::prefixSum(ld.topicLikelihood.data(), ld.topicLikelihood.size());
			return &ld.topicLikelihood[0];
		}

		template<int INC>
		inline void addOnlyWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, Vid vid, Tid tid) const
		{
			assert(tid < ld.numTableByTopic.size());
			assert(vid < this->realV);

			if (INC > 0 && tid >= doc.numByTopic.size())
			{
				size_t oldSize = doc.numByTopic.size();
				doc.numByTopic.conservativeResize(tid + 1);
				doc.numByTopic.tail(tid + 1 - oldSize).setZero();
			}
			constexpr bool DEC = INC < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
		}

		template<int INC> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, Vid vid, size_t tableId, Tid tid) const
		{
			addOnlyWordTo<INC>(ld, doc, pid, vid, tid);
			constexpr bool DEC = INC < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			if (INC < 0) assert(doc.numTopicByTable[tableId].num > 0);
			updateCnt<DEC>(doc.numTopicByTable[tableId].num, INC * weight);
			if (INC < 0 && !doc.numTopicByTable[tableId])  // deleting table
			{
				size_t topic = doc.numTopicByTable[tableId].topic;
				updateCnt<DEC>(ld.numTableByTopic[topic], INC);
				ld.totalTable += INC;

				if (!ld.numTableByTopic[topic]) // delete topic
				{
					//printf("Deleted Topic #%zd\n", topic);
				}
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.numTopicByTable[doc.Zs[w]].topic);
				calcWordTopicProb(ld, doc.words[w]);
				auto topicDist = getTopicLikelihoods(ld);
				auto dist = getTableLikelihoods(ld, doc, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + doc.numTopicByTable.size() + (_infer ? 0 : 1), rgs);
				if (doc.Zs[w] == doc.numTopicByTable.size()) // create new table
				{
					size_t K = ld.numByTopic.size();
					Tid newTopic = sample::sampleFromDiscreteAcc(topicDist, topicDist + K + 1, rgs);
					if (newTopic == K) // create new topic
					{
						newTopic = addTopic(ld);
						//printf("New Topic #%zd\n", newTopic);
					}
					doc.Zs[w] = doc.addNewTable(newTopic);
					++ld.numTableByTopic[newTopic];
					++ld.totalTable;
				}
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.numTopicByTable[doc.Zs[w]].topic);
			}

			for (size_t t = 0; t < doc.getNumTable(); ++t)
			{
				auto& curTable = doc.numTopicByTable[t];
				if (!curTable) continue;
				--ld.numTableByTopic[curTable.topic];
				size_t K = ld.numByTopic.size();
				ld.zLikelihood.resize(K + 1);
				ld.zLikelihood.setZero();
				for (size_t w = 0; w < doc.words.size(); ++w)
				{
					if (doc.words[w] >= this->realV) continue;
					if (doc.Zs[w] != t) continue;
					addOnlyWordTo<-1>(ld, doc, w, doc.words[w], curTable.topic);
					ld.zLikelihood.head(K).array() += ((ld.numByTopicWord.col(doc.words[w]).array().template cast<Float>() + this->eta)
						/ (ld.numByTopic.array().template cast<Float>() + this->realV * this->eta)).log();
					ld.zLikelihood[K] += log(1. / this->realV);
				}
				ld.zLikelihood = (ld.zLikelihood.array() - ld.zLikelihood.maxCoeff()).exp();
				auto topicDist = getTopicLikelihoods(ld);
				Tid newTopic = sample::sampleFromDiscreteAcc(topicDist, topicDist + K + 1, rgs);
				if (newTopic == K) // create new topic
				{
					newTopic = addTopic(ld);
					//printf("New Topic #%zd\n", newTopic);
				}
				curTable.topic = newTopic;
				for (size_t w = 0; w < doc.words.size(); ++w)
				{
					if (doc.words[w] >= this->realV) continue;
					if (doc.Zs[w] != t) continue;
					addOnlyWordTo<1>(ld, doc, w, doc.words[w], curTable.topic);
				}
				++ld.numTableByTopic[curTable.topic];
			}
		}

		void updateGlobalInfo(ThreadPool& pool, _ModelState* localData)
		{
			std::vector<std::future<void>> res;
			auto& K = this->K;
			K = 0;
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				K = std::max(K, (Tid)localData[i].numByTopic.size());
			}

			// synchronize topic size of all documents
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res.emplace_back(pool.enqueue([&, this](size_t threadId, size_t b, size_t e)
				{
					for (size_t j = b; j < e; ++j)
					{
						auto& doc = this->docs[j];
						if (doc.numByTopic.size() >= K) continue;
						size_t oldSize = doc.numByTopic.size();
						doc.numByTopic.conservativeResize(K);
						doc.numByTopic.tail(K - oldSize).setZero();
					}
				}, this->docs.size() * i / pool.getNumWorkers(), this->docs.size() * (i + 1) / pool.getNumWorkers()));
			}
			for (auto& r : res) r.get();
		}

		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, RandGen*, const _ExtraDocData& edd) const
		{
			std::vector<std::future<void>> res;
			const size_t V = this->realV;
			auto K = this->K;

			if (K > globalState.numByTopic.size())
			{
				size_t oldSize = globalState.numByTopic.size();
				globalState.numByTopic.conservativeResize(K);
				globalState.numByTopic.tail(K - oldSize).setZero();
				globalState.numTableByTopic.conservativeResize(K);
				globalState.numTableByTopic.tail(K - oldSize).setZero();
				globalState.numByTopicWord.conservativeResize(K, Eigen::NoChange);
				globalState.numByTopicWord.block(oldSize, 0, K - oldSize, V).setZero();
			}

			tState = globalState;
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				size_t locK = localData[i].numByTopic.size();
				globalState.numByTopic.head(locK) 
					+= localData[i].numByTopic.head(locK) - tState.numByTopic.head(locK);
				globalState.numTableByTopic.head(locK)
					+= localData[i].numTableByTopic.head(locK) - tState.numTableByTopic.head(locK);
				globalState.numByTopicWord.block(0, 0, locK, V)
					+= localData[i].numByTopicWord.block(0, 0, locK, V) - tState.numByTopicWord.block(0, 0, locK, V);
			}

			// make all count being positive
			if (_tw != TermWeight::one)
			{
				globalState.numByTopic = globalState.numByTopic.cwiseMax(0);
				globalState.numByTopicWord = globalState.numByTopicWord.cwiseMax(0);
			}

			globalState.totalTable = accumulate(this->docs.begin(), this->docs.end(), 0, [](size_t sum, const _DocType& doc)
			{
				return sum + doc.getNumTable();
			});
			
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res.emplace_back(pool.enqueue([&, this, i](size_t threadId)
				{
					localData[i] = globalState;
				}));
			}
			for (auto& r : res) r.get();
		}

		/* this LL calculation is based on https://github.com/blei-lab/hdp/blob/master/hdp/state.cpp */
		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const auto alpha = this->alpha;
			double ll = 0;

			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll += doc.getNumTable() * log(alpha) - math::lgammaT(doc.getSumWordWeight() + alpha) + math::lgammaT(alpha);
				for (auto& nt : doc.numTopicByTable)
				{
					if (nt) ll += math::lgammaT(nt.num);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			const size_t V = this->realV;
			const auto K = this->K;
			const auto eta = this->eta;
			double ll = 0;
			// table partition ll
			size_t liveK = 0;
			for (Tid k = 0; k < K; ++k)
			{
				if (!isLiveTopic(k)) continue;
				ll += math::lgammaT(ld.numTableByTopic[k]);
				++liveK;
			}
			ll += liveK * log(gamma) - math::lgammaT(ld.totalTable + gamma) + math::lgammaT(gamma);
			// topic word ll
			ll += liveK * math::lgammaT(V * eta);
			for (Tid k = 0; k < K; ++k)
			{
				if (!isLiveTopic(k)) continue;
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (Vid v = 0; v < V; ++v)
				{
					if (!ld.numByTopicWord(k, v)) continue;
					ll += math::lgammaT(ld.numByTopicWord(k, v) + eta) - math::lgammaT(eta);
				}
			}
			return ll;
		}


		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			const auto K = this->K;
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				this->globalState.numTableByTopic = Eigen::Matrix<int32_t, -1, 1>::Zero(K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
			}
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			doc.numByTopic.init(topicDocPtr, this->K);
			doc.Zs = tvector<Tid>(wordSize);
			if (_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		template<bool _Infer>
		void updateStateWithDoc(typename BaseClass::Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			if (doc.getNumTable() == 0)
			{
				Tid k = g.theta(rgs);
				doc.addNewTable(k);
				++ld.numTableByTopic[k];
				++ld.totalTable;
			}
			doc.Zs[i] = 0;
			addWordTo<1>(ld, doc, i, doc.words[i], 0, doc.numTopicByTable[0].topic);
		}

		std::vector<size_t> _getTopicsCount() const
		{
			std::vector<size_t> cnt(this->K);
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.numTopicByTable[doc.Zs[i]].topic];
				}
			}
			return cnt;
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, gamma);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, gamma);

		HDPModel(size_t initialK = 2, Float _alpha = 0.1, Float _eta = 0.01, Float _gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(initialK, _alpha, _eta, _rg), gamma(_gamma)
		{
			if (_gamma <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong gamma value (gamma = %f)", _gamma));
		}

		size_t getTotalTables() const override
		{
			return accumulate(this->docs.begin(), this->docs.end(), 0, [](size_t sum, const _DocType& doc)
			{
				return sum + doc.getNumTable();
			});
		}

		size_t getLiveK() const override
		{ 
			return this->globalState.numTableByTopic.count();
		}

		GETTER(Gamma, Float, gamma);

		bool isLiveTopic(Tid tid) const override
		{
			return this->globalState.numTableByTopic[tid];
		}

		void setWordPrior(const std::string& word, const std::vector<Float>& priors) override
		{
			THROW_ERROR_WITH_INFO(exception::Unimplemented, "HDPModel doesn't provide setWordPrior function.");
		}

		std::vector<Float> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Matrix<Float, -1, 1>> { ret.data(), this->K }.array() = doc.numByTopic.array().template cast<Float>() / doc.getSumWordWeight();
			return ret;
		}
	};

	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentHDP<_tw>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		this->numByTopic.init(ptr, mdl.getK());
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numTopicByTable[this->Zs[i]].num += _tw != TermWeight::one ? this->wordWeights[i] : 1;
			this->numByTopic[numTopicByTable[this->Zs[i]].topic] += _tw != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}