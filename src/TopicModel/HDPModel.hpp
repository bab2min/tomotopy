#pragma once
#include "LDAModel.hpp"

/*
Implementation of HDP using Gibbs sampling by bab2min

* Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2005). Sharing clusters among related groups: Hierarchical Dirichlet processes. In Advances in neural information processing systems (pp. 1385-1392).
* Newman, D., Asuncion, A., Smyth, P., & Welling, M. (2009). Distributed algorithms for topic models. Journal of Machine Learning Research, 10(Aug), 1801-1828.
*/

namespace tomoto
{
	template<TermWeight _TW>
	struct DocumentHDP : public DocumentLDA<_TW>
	{
		/* 
		for DocumentHDP, the topic in numByTopic, Zs indicates 'table id', not 'topic id'.
		to get real 'topic id', check the topic field of numTopicByTable.
		*/
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;
		struct TableTopicInfo
		{
			WeightType num = 0;
			TID topic = 0;
			operator const bool() const
			{
				return num > (WeightType)1e-2;
			}

			void serializerWrite(std::ostream& writer) const
			{
				serializer::writeMany(writer, topic);
			}

			void serializerRead(std::istream& reader)
			{
				serializer::readMany(reader, topic);
			}
		};
		std::vector<TableTopicInfo> numTopicByTable;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, numTopicByTable);

		size_t getNumTable() const
		{
			return std::count_if(numTopicByTable.begin(), numTopicByTable.end(), [](const auto& e) { return (bool)e; });
		}
		size_t addNewTable(TID tid)
		{
			return insertIntoEmpty(numTopicByTable, TableTopicInfo{ 0, tid });
		}

		void update(WeightType* ptr, size_t K)
		{
			DocumentLDA<_TW>::update(ptr, K);
			for (size_t i = 0; i < this->Zs.size(); ++i)
			{
				numTopicByTable[this->Zs[i]].num += _TW != TermWeight::one ? this->wordWeights[i] : 1;
			}
		}
	};

	template<TermWeight _TW>
	struct ModelStateHDP : public ModelStateLDA<_TW>
	{
		Eigen::Matrix<FLOAT, -1, 1> tableLikelihood, topicLikelihood;
		Eigen::Matrix<int32_t, -1, 1> numTableByTopic;
		size_t totalTable = 0;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_TW>, numTableByTopic, totalTable);
	};

	class IHDPModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentHDP<TermWeight::one>;
		static IHDPModel* create(TermWeight _weight, size_t _K = 1, FLOAT alpha = 0.1, FLOAT eta = 0.01, FLOAT gamma = 0.1, const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual FLOAT getGamma() const = 0;
		virtual size_t getTotalTables() const = 0;
		virtual size_t getLiveK() const = 0;
		virtual bool isLiveTopic(TID tid) const = 0;
	};

	template<TermWeight _TW,
		typename _Interface = IHDPModel,
		typename _Derived = void,
		typename _DocType = DocumentHDP<_TW>,
		typename _ModelState = ModelStateHDP<_TW>>
	class HDPModel : public LDAModel<_TW, false, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, false, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		FLOAT gamma;
		size_t addTopic(_ModelState& ld, VID vid) const
		{
			const size_t V = this->dict.size();
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

		FLOAT* getWordTopicProb(_ModelState& ld, VID vid) const
		{
			const size_t V = this->dict.size();
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);
			return &zLikelihood[0];
		}

		FLOAT* getTableLikelihoods(_ModelState& ld, _DocType& doc, VID vid) const
		{
			const size_t V = this->dict.size();
			assert(vid < V);
			const size_t T = doc.numTopicByTable.size();
			const auto K = ld.numByTopic.size();
			FLOAT acc = 0;
			ld.tableLikelihood.resize(T + 1);
			for (size_t t = 0; t < T; ++t)
			{
				ld.tableLikelihood[t] = acc += doc.numTopicByTable[t].num * ld.zLikelihood[doc.numTopicByTable[t].topic];
			}
			FLOAT pNewTable = ld.topicLikelihood[K] / (gamma + ld.totalTable);
			ld.tableLikelihood[T] = acc += pNewTable * this->alpha;
			return &ld.tableLikelihood[0];
		}

		FLOAT* getTopicLikelihoods(_ModelState& ld, VID vid) const
		{
			const size_t V = this->dict.size();
			assert(vid < V);
			const auto K = ld.numByTopic.size();
			ld.topicLikelihood.resize(K + 1);
			ld.topicLikelihood.head(K) = ld.zLikelihood.array().template cast<FLOAT>() * ld.numTableByTopic.array().template cast<FLOAT>();
			ld.topicLikelihood[K] = gamma / V;
			sample::prefixSum(ld.topicLikelihood.data(), ld.topicLikelihood.size());
			return &ld.topicLikelihood[0];
		}

		template<int INC> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, size_t tableId, TID tid) const
		{
			assert(tid < ld.numTableByTopic.size());
			const size_t V = this->dict.size();
			assert(vid < V);

			if (INC > 0 && tid >= doc.numByTopic.size())
			{
				size_t oldSize = doc.numByTopic.size();
				doc.numByTopic.conservativeResize(tid + 1);
				doc.numByTopic.tail(tid + 1 - oldSize).setZero();
			}
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			typename std::conditional<_TW != TermWeight::one, float, int32_t>::type weight
				= _TW != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<DEC>(doc.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopic[tid], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(tid, vid), INC * weight);
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

		void sampleDocument(_DocType& doc, _ModelState& ld, RANDGEN& rgs) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.numTopicByTable[doc.Zs[w]].topic);
				getWordTopicProb(ld, doc.words[w]);
				auto topicDist = getTopicLikelihoods(ld, doc.words[w]);
				auto dist = getTableLikelihoods(ld, doc, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + doc.numTopicByTable.size() + 1, rgs);
				if (doc.Zs[w] == doc.numTopicByTable.size()) // create new table
				{
					size_t K = ld.numByTopic.size();
					TID newTopic = sample::sampleFromDiscreteAcc(topicDist, topicDist + K + 1, rgs);
					if (newTopic == K) // create new topic
					{
						newTopic = addTopic(ld, doc.words[w]);
						//printf("New Topic #%zd\n", newTopic);
					}
					doc.Zs[w] = doc.addNewTable(newTopic);
					++ld.numTableByTopic[newTopic];
					++ld.totalTable;
				}
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.numTopicByTable[doc.Zs[w]].topic);
			}
		}

		void updateGlobal(ThreadPool& pool, _ModelState* localData)
		{
			std::vector<std::future<void>> res(pool.getNumWorkers());
			const size_t V = this->dict.size();
			auto& K = this->K;
			K = 0;
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				K = std::max(K, (TID)localData[i].numByTopic.size());
			}

			if (K > this->globalState.numByTopic.size())
			{
				size_t oldSize = this->globalState.numByTopic.size();
				this->globalState.numByTopic.conservativeResize(K);
				this->globalState.numByTopic.tail(K - oldSize).setZero();
				this->globalState.numTableByTopic.conservativeResize(K);
				this->globalState.numTableByTopic.tail(K - oldSize).setZero();
				this->globalState.numByTopicWord.conservativeResize(K, Eigen::NoChange);
				this->globalState.numByTopicWord.block(oldSize, 0, K - oldSize, V).setZero();
			}

			// synchronize topic size of all documents
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, this](size_t threadId, size_t b, size_t e)
				{
					for (size_t j = b; j < e; ++j)
					{
						auto& doc = this->docs[j];
						if (doc.numByTopic.size() >= K) continue;
						size_t oldSize = doc.numByTopic.size();
						doc.numByTopic.conservativeResize(K);
						doc.numByTopic.tail(K - oldSize).setZero();
					}
				}, this->docs.size() * i / pool.getNumWorkers(), this->docs.size() * (i + 1) / pool.getNumWorkers());
			}
			for (auto&& r : res) r.get();

			this->tState = this->globalState;
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				size_t locK = localData[i].numByTopic.size();
				this->globalState.numByTopic.head(locK) 
					+= localData[i].numByTopic.head(locK) - this->tState.numByTopic.head(locK);
				this->globalState.numTableByTopic.head(locK)
					+= localData[i].numTableByTopic.head(locK) - this->tState.numTableByTopic.head(locK);
				this->globalState.numByTopicWord.block(0, 0, locK, V)
					+= localData[i].numByTopicWord.block(0, 0, locK, V) - this->tState.numByTopicWord.block(0, 0, locK, V);
			}

			// make all count being positive
			if (_TW != TermWeight::one)
			{
				this->globalState.numByTopic = this->globalState.numByTopic.cwiseMax(0);
				this->globalState.numByTopicWord = this->globalState.numByTopicWord.cwiseMax(0);
			}

			this->globalState.totalTable = accumulate(this->docs.begin(), this->docs.end(), 0, [](size_t sum, auto doc)
			{
				return sum + doc.getNumTable();
			});
			
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				res[i] = pool.enqueue([&, this, i](size_t threadId)
				{
					localData[i] = this->globalState;
				});
			}
			for (auto&& r : res) r.get();
		}

		/* this LL calculation is based on https://github.com/blei-lab/hdp/blob/master/hdp/state.cpp */
		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			const size_t V = this->dict.size();
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			double ll = 0;

			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				ll += doc.getNumTable() * log(alpha) - math::lgammaT(doc.template getSumWordWeight<_TW>() + alpha) + math::lgammaT(alpha);
				for (auto&& nt : doc.numTopicByTable)
				{
					if (nt) ll += math::lgammaT(nt.num);
				}
			}
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			const size_t V = this->dict.size();
			const auto K = this->K;
			const auto alpha = this->alpha;
			const auto eta = this->eta;
			double ll = 0;
			// table partition ll
			size_t liveK = 0;
			for (TID k = 0; k < K; ++k)
			{
				if (!isLiveTopic(k)) continue;
				ll += math::lgammaT(ld.numTableByTopic[k]);
				++liveK;
			}
			ll += liveK * log(gamma) - math::lgammaT(ld.totalTable + gamma) + math::lgammaT(gamma);
			// topic word ll
			ll += liveK * math::lgammaT(V * eta);
			for (TID k = 0; k < K; ++k)
			{
				if (!isLiveTopic(k)) continue;
				ll -= math::lgammaT(ld.numByTopic[k] + V * eta);
				for (VID v = 0; v < V; ++v)
				{
					if (!ld.numByTopicWord(k, v)) continue;
					ll += math::lgammaT(ld.numByTopicWord(k, v) + eta) - math::lgammaT(eta);
				}
			}
			return ll;
		}


		void initGlobalState(bool initDocs)
		{
			const size_t V = this->dict.size();
			const auto K = this->K;
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(K);
				this->globalState.numTableByTopic = Eigen::Matrix<int32_t, -1, 1>::Zero(K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
			}
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			doc.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
			doc.Zs = tvector<TID>(wordSize);
			if (_TW != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		void updateStateWithDoc(typename BaseClass::Generator& g, _ModelState& ld, RANDGEN& rgs, _DocType& doc, size_t i) const
		{
			if (i == 0)
			{
				TID k = g.theta(rgs);
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
				for (auto z : doc.Zs) ++cnt[doc.numTopicByTable[z].topic];
			}
			return cnt;
		}

	public:
		HDPModel(size_t initialK = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, FLOAT _gamma = 0.1, const RANDGEN& _rg = RANDGEN{ std::random_device{}() })
			: BaseClass(initialK, _alpha, _eta, _rg), gamma(_gamma)
		{}

		size_t getTotalTables() const override
		{
			return accumulate(this->docs.begin(), this->docs.end(), 0, [](size_t sum, auto doc)
			{
				return sum + doc.getNumTable();
			});
		}

		size_t getLiveK() const override
		{ 
			return this->globalState.numTableByTopic.count();
		}

		GETTER(Gamma, FLOAT, gamma);

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, gamma);

		bool isLiveTopic(TID tid) const override
		{
			return this->globalState.numTableByTopic[tid];
		}
	};

	IHDPModel* IHDPModel::create(TermWeight _weight, size_t _K, FLOAT _alpha , FLOAT _eta, FLOAT _gamma, const RANDGEN& _rg)
	{
		SWITCH_TW(_weight, HDPModel, _K, _alpha, _eta, _gamma, _rg);
	}
}