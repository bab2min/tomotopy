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
		Vector tableLikelihood, topicLikelihood;
		Eigen::Matrix<int32_t, -1, 1> numTableByTopic;
		size_t totalTable = 0;

		DEFINE_SERIALIZER_AFTER_BASE(ModelStateLDA<_tw>, numTableByTopic, totalTable);
	};

	template<TermWeight _tw, typename _RandGen,
		typename _Interface = IHDPModel,
		typename _Derived = void,
		typename _DocType = DocumentHDP<_tw>,
		typename _ModelState = ModelStateHDP<_tw>>
	class HDPModel : public LDAModel<_tw, _RandGen, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, HDPModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Float gamma;

		template<typename _NumFunc>
		static Float estimateConcentrationParameter(_NumFunc ns, Float tableCnt, size_t size, Float alpha, _RandGen& rgs)
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

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
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
			for (pos = 0; pos < (size_t)ld.numTableByTopic.size(); ++pos)
			{
				if (!ld.numTableByTopic[pos]) break;
			}
			
			if (pos >= (size_t)ld.numByTopic.size())
			{
				size_t oldSize = ld.numByTopic.size(), newSize = pos + 1;
				ld.numTableByTopic.conservativeResize(newSize);
				ld.numTableByTopic.tail(newSize - oldSize).setZero();
				ld.numByTopic.conservativeResize(newSize);
				ld.numByTopic.tail(newSize - oldSize).setZero();
				ld.numByTopicWord.conservativeResize(newSize, V);
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

		template<int _inc>
		inline void addOnlyWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, Vid vid, Tid tid) const
		{
			assert(tid < ld.numTableByTopic.size());
			assert(vid < this->realV);

			if (_inc > 0 && tid >= doc.numByTopic.size())
			{
				size_t oldSize = doc.numByTopic.size();
				doc.numByTopic.conservativeResize(tid + 1, 1);
				doc.numByTopic.tail(tid + 1 - oldSize).setZero();
			}
			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			updateCnt<_dec>(doc.numByTopic[tid], _inc * weight);
			updateCnt<_dec>(ld.numByTopic[tid], _inc * weight);
			updateCnt<_dec>(ld.numByTopicWord(tid, vid), _inc * weight);
		}

		template<int _inc> 
		inline void addWordTo(_ModelState& ld, _DocType& doc, size_t pid, Vid vid, size_t tableId, Tid tid) const
		{
			addOnlyWordTo<_inc>(ld, doc, pid, vid, tid);
			constexpr bool _dec = _inc < 0 && _tw != TermWeight::one;
			typename std::conditional<_tw != TermWeight::one, float, int32_t>::type weight
				= _tw != TermWeight::one ? doc.wordWeights[pid] : 1;

			if (_inc < 0) assert(doc.numTopicByTable[tableId].num > 0);
			updateCnt<_dec>(doc.numTopicByTable[tableId].num, _inc * weight);
			if (_inc < 0 && !doc.numTopicByTable[tableId])  // deleting table
			{
				size_t topic = doc.numTopicByTable[tableId].topic;
				updateCnt<_dec>(ld.numTableByTopic[topic], _inc);
				ld.totalTable += _inc;

				if (!ld.numTableByTopic[topic]) // delete topic
				{
					//printf("Deleted Topic #%zd\n", topic);
				}
			}
		}

		template<ParallelScheme _ps, bool _infer, typename _ExtraDocData>
		void sampleDocument(_DocType& doc, const _ExtraDocData& edd, size_t docId, _ModelState& ld, _RandGen& rgs, size_t iterationCnt, size_t partitionId = 0) const
		{
			// sample a table for each word
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w], doc.numTopicByTable[doc.Zs[w]].topic);
				calcWordTopicProb(ld, doc.words[w]);
				auto topicDist = getTopicLikelihoods(ld);
				auto dist = getTableLikelihoods(ld, doc, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + doc.numTopicByTable.size() + 1, rgs);
				if (doc.Zs[w] == doc.numTopicByTable.size()) // create new table
				{
					size_t K = ld.numByTopic.size();
					Tid newTopic = sample::sampleFromDiscreteAcc(topicDist, topicDist + K + (_infer ? 0 : 1), rgs);
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

			// sample a topic for each table
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

				// turn off dead topics
				for (size_t k = 0; k < K; ++k)
				{
					if (!ld.numTableByTopic[k]) ld.zLikelihood[k] = -INFINITY;
				}

				ld.zLikelihood = (ld.zLikelihood.array() - ld.zLikelihood.maxCoeff()).exp();
				auto topicDist = getTopicLikelihoods(ld);
				Tid newTopic = sample::sampleFromDiscreteAcc(topicDist, topicDist + K + (_infer ? 0 : 1), rgs);
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
						doc.numByTopic.conservativeResize(K, 1);
						doc.numByTopic.tail(K - oldSize).setZero();
					}
				}, this->docs.size() * i / pool.getNumWorkers(), this->docs.size() * (i + 1) / pool.getNumWorkers()));
			}
			for (auto& r : res) r.get();
		}

		template<ParallelScheme _ps, typename _ExtraDocData>
		void mergeState(ThreadPool& pool, _ModelState& globalState, _ModelState& tState, _ModelState* localData, _RandGen*, const _ExtraDocData& edd) const
		{
			const size_t V = this->realV;
			auto K = this->K;

			if (K > globalState.numByTopic.size())
			{
				size_t oldSize = globalState.numByTopic.size();
				globalState.numByTopic.conservativeResize(K);
				globalState.numByTopic.tail(K - oldSize).setZero();
				globalState.numTableByTopic.resize(K);
				globalState.numByTopicWord.conservativeResize(K, V);
				globalState.numByTopicWord.block(oldSize, 0, K - oldSize, V).setZero();
			}

			tState = globalState;
			for (size_t i = 0; i < pool.getNumWorkers(); ++i)
			{
				size_t locK = localData[i].numByTopic.size();
				globalState.numByTopic.head(locK) 
					+= localData[i].numByTopic.head(locK) - tState.numByTopic.head(locK);
				globalState.numByTopicWord.block(0, 0, locK, V)
					+= localData[i].numByTopicWord.block(0, 0, locK, V) - tState.numByTopicWord.block(0, 0, locK, V);
			}

			// make all count being positive
			if (_tw != TermWeight::one)
			{
				globalState.numByTopic = globalState.numByTopic.cwiseMax(0);
				globalState.numByTopicWord.matrix() = globalState.numByTopicWord.cwiseMax(0);
			}


			globalState.numTableByTopic.setZero();
			for (auto& doc : this->docs)
			{
				for (auto& table : doc.numTopicByTable)
				{
					if (table) globalState.numTableByTopic[table.topic]++;
				}
			}
			globalState.totalTable = globalState.numTableByTopic.sum();
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
			size_t liveK = (ld.numTableByTopic.array() > 0).template cast<size_t>().sum();
			Eigen::ArrayXf lg = math::lgammaApprox(ld.numTableByTopic.array().template cast<Float>());
			ll += (ld.numTableByTopic.array() > 0).select(lg, 0).sum();
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
				//this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(K, V);
				this->globalState.numByTopicWord.init(nullptr, K, V);
			}
		}

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			doc.numByTopic.init(nullptr, this->K, 1);
			doc.numTopicByTable.clear();
			doc.Zs = tvector<Tid>(wordSize, non_topic_id);
			if (_tw != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		template<bool _infer>
		void updateStateWithDoc(typename BaseClass::Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			Tid t;
			std::vector<double> dist;
			dist.emplace_back(this->alpha);
			for (auto& d : doc.numTopicByTable) dist.emplace_back(d.num);
			std::discrete_distribution<Tid> ddist{ dist.begin(), dist.end() };
			t = ddist(rgs);
			if (t == 0)
			{
				// new table
				Tid k;
				if (_infer)
				{
					std::uniform_int_distribution<> theta{ 0, this->K - 1 };
					do
					{
						k = theta(rgs);
					} while (!isLiveTopic(k));
				}
				else
				{
					k = g.theta(rgs);
				}
				t = doc.addNewTable(k);
				++ld.numTableByTopic[k];
				++ld.totalTable;
			}
			else
			{
				t -= 1;
			}
			doc.Zs[i] = t;
			addWordTo<1>(ld, doc, i, doc.words[i], doc.Zs[i], doc.numTopicByTable[doc.Zs[i]].topic);
		}

		std::vector<uint64_t> _getTopicsCount() const
		{
			std::vector<uint64_t> cnt(this->K);
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

		HDPModel(const HDPArgs& args)
			: BaseClass(args), gamma(args.gamma)
		{
			if (gamma <= 0) THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong gamma value (gamma = %f)", gamma));
			if (args.alpha.size() > 1) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "Asymmetric alpha is not supported at HDP.");
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
			THROW_ERROR_WITH_INFO(exc::Unimplemented, "HDPModel doesn't provide setWordPrior function.");
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K);
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
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

		std::unique_ptr<ILDAModel> convertToLDA(float topicThreshold, std::vector<Tid>& newK) const override
		{
			auto cnt = _getTopicsCount();
			std::vector<std::pair<uint64_t, size_t>> cntIdx;
			float sum = (float)std::accumulate(cnt.begin(), cnt.end(), 0);
			for (size_t i = 0; i < cnt.size(); ++i)
			{
				cntIdx.emplace_back(cnt[i], i);
			}
			std::sort(cntIdx.rbegin(), cntIdx.rend());

			size_t liveK = 0;
			newK.clear();
			newK.resize(cntIdx.size(), -1);
			for (size_t i = 0; i < cntIdx.size(); ++i)
			{
				if (i && cntIdx[i].first / sum <= topicThreshold) break;
				newK[cntIdx[i].second] = (Tid)i;
				liveK++;
			}

			LDAArgs args;
			args.k = liveK;
			args.alpha[0] = 0.1f;
			args.eta = this->eta;
			auto lda = std::make_unique<LDAModel<_tw, _RandGen>>(args);
			lda->dict = this->dict;
			
			for (auto& doc : this->docs)
			{
				auto d = lda->_makeFromRawDoc(doc);
				lda->_addDoc(d);
			}

			lda->realV = this->realV;
			lda->realN = this->realN;
			lda->weightedN = this->weightedN;
			lda->prepare(true, 0, 0, 0, false);

			auto selectFirst = [&](const std::pair<size_t, size_t>& p) { return std::max(p.first / sum - topicThreshold, 0.f); };
			std::discrete_distribution<size_t> randomTopic{
				makeTransformIter(cntIdx.begin(), selectFirst),
				makeTransformIter(cntIdx.end(), selectFirst) 
			};
			
			std::mt19937_64 rng;

			for (size_t i = 0; i < this->docs.size(); ++i)
			{
				for (size_t j = 0; j < this->docs[i].Zs.size(); ++j)
				{
					if (this->docs[i].Zs[j] == non_topic_id)
					{
						lda->docs[i].Zs[j] = non_topic_id;
						continue;
					}
					Tid newTopic = newK[this->docs[i].numTopicByTable[this->docs[i].Zs[j]].topic];
					while (newTopic == (Tid)-1) newTopic = newK[randomTopic(rng)];
					lda->docs[i].Zs[j] = newTopic;
				}
			}

			lda->resetStatistics();
			lda->optimizeParameters(*(ThreadPool*)nullptr, nullptr, nullptr);

			return lda;
		}

		std::vector<Tid> purgeDeadTopics() override
		{
			std::vector<Tid> relocation(this->K, -1);
			Tid numLiveTopics = 0;
			for (size_t i = 0; i < this->K; ++i)
			{
				if (this->globalState.numTableByTopic[i])
				{
					relocation[i] = numLiveTopics++;
				}
			}

			for (auto& doc : this->docs)
			{
				for (auto& nt : doc.numTopicByTable)
				{
					nt.topic = (relocation[nt.topic] == (Tid)-1) ? 0 : relocation[nt.topic];
				}

				for (size_t i = 0; i < relocation.size(); ++i)
				{
					if (relocation[i] == (Tid)-1) continue;
					doc.numByTopic[relocation[i]] = doc.numByTopic[i];
				}
				doc.numByTopic.conservativeResize(numLiveTopics, 1);
			}

			for (auto tt : { &this->globalState, &this->tState })
			{
				auto& numTableByTopic = tt->numTableByTopic;
				auto& numByTopic = tt->numByTopic;
				auto& numByTopicWord = tt->numByTopicWord;
				for (size_t i = 0; i < relocation.size(); ++i)
				{
					if (relocation[i] == (Tid)-1) continue;
					numTableByTopic[relocation[i]] = numTableByTopic[i];
					numByTopic[relocation[i]] = numByTopic[i];
					numByTopicWord.row(relocation[i]) = numByTopicWord.row(i);
				}
				numTableByTopic.conservativeResize(numLiveTopics);
				numByTopic.conservativeResize(numLiveTopics);
				numByTopicWord.conservativeResize(numLiveTopics, numByTopicWord.cols());
			}
			this->K = numLiveTopics;
			return relocation;
		}
	};

	template<TermWeight _tw>
	template<typename _TopicModel>
	void DocumentHDP<_tw>::update(WeightType * ptr, const _TopicModel & mdl)
	{
		this->numByTopic.init(ptr, mdl.getK(), 1);
		for (size_t i = 0; i < this->Zs.size(); ++i)
		{
			if (this->words[i] >= mdl.getV()) continue;
			numTopicByTable[this->Zs[i]].num += _tw != TermWeight::one ? this->wordWeights[i] : 1;
			this->numByTopic[numTopicByTable[this->Zs[i]].topic] += _tw != TermWeight::one ? this->wordWeights[i] : 1;
		}
	}
}
