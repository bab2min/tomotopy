#pragma once
#include "../Utils/Utils.hpp"
#include "../Utils/Dictionary.h"
#include "../Utils/tvector.hpp"
#include "../Utils/ThreadPool.hpp"
#include "../Utils/serializer.hpp"
#include "../Utils/exception.h"


namespace tomoto
{
#if _WIN32 || _WIN64
#if _WIN64
	typedef std::mt19937_64 RandGen;
#else
	typedef std::mt19937 RandGen;
#endif
#endif

#if __GNUC__
#if __x86_64__ || __ppc64__
	typedef std::mt19937_64 RandGen;
#else
	typedef std::mt19937 RandGen;
#endif
#endif

	class DocumentBase
	{
	public:
		FLOAT weight = 1;
		tvector<VID> words; // word id of each word
		std::vector<uint32_t> wOrder; // original word order (optional)
		DocumentBase(FLOAT _weight = 1) : weight(_weight) {}
		virtual ~DocumentBase() {}

		DEFINE_SERIALIZER(serializer::MagicConstant("Document"), weight, words, wOrder);
	};

	enum class ParallelScheme { default_, none, copy_merge, partition, size };

	inline const char* toString(ParallelScheme ps)
	{
		switch (ps)
		{
		case ParallelScheme::default_: return "default";
		case ParallelScheme::none: return "none";
		case ParallelScheme::copy_merge: return "copy_merge";
		case ParallelScheme::partition: return "partition";
		default: return "unknown";
		}
	}

	class ITopicModel
	{
	public:
		virtual void saveModel(std::ostream& writer, bool fullModel) const = 0;
		virtual void loadModel(std::istream& reader) = 0;
		virtual const DocumentBase* getDoc(size_t docId) const = 0;

		virtual double getLLPerWord() const = 0;
		virtual double getPerplexity() const = 0;
		virtual size_t getV() const = 0;
		virtual size_t getN() const = 0;
		virtual size_t getNumDocs() const = 0;
		virtual const Dictionary& getVocabDict() const = 0;
		virtual const std::vector<size_t>& getVocabFrequencies() const = 0;

		virtual int train(size_t iteration, size_t numWorkers, ParallelScheme ps = ParallelScheme::default_) = 0;
		virtual void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t removeTopN = 0) = 0;
		virtual std::vector<FLOAT> getWidsByTopic(TID tid) const = 0;
		virtual std::vector<std::pair<std::string, FLOAT>> getWordsByTopicSorted(TID tid, size_t topN) const = 0;

		virtual std::vector<std::pair<std::string, FLOAT>> getWordsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		
		virtual std::vector<FLOAT> getTopicsByDoc(const DocumentBase* doc) const = 0;
		virtual std::vector<std::pair<TID, FLOAT>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		virtual std::vector<double> infer(const std::vector<DocumentBase*>& docs, size_t maxIter, FLOAT tolerance, size_t numWorkers, ParallelScheme ps, bool together) const = 0;
		virtual ~ITopicModel() {}
	};

	template<class _TyKey, class _TyValue>
	static std::vector<std::pair<_TyKey, _TyValue>> extractTopN(const std::vector<_TyValue>& vec, size_t topN)
	{
		typedef std::pair<_TyKey, _TyValue> pair_t;
		std::vector<pair_t> ret;
		_TyKey k = 0;
		for (auto& t : vec)
		{
			ret.emplace_back(std::make_pair(k++, t));
		}
		std::sort(ret.begin(), ret.end(), [](const pair_t& a, const pair_t& b)
		{
			return a.second > b.second;
		});
		if (topN < ret.size()) ret.erase(ret.begin() + topN, ret.end());
		return ret;
	}

	namespace flags
	{
		enum
		{
			continuous_doc_data = 1 << 0,
			shared_state = 1 << 1,
			partitioned_multisampling = 1 << 2,
			end_flag_of_TopicModel = 1 << 3,
		};
	}

	template<size_t _Flags, typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
	class TopicModel : public _Interface
	{
		friend class Document;
	public:
		using DocType = _DocType;
	protected:
		RandGen rg;
		std::vector<VID> words;
		std::vector<uint32_t> wOffsetByDoc;

		std::vector<DocType> docs;
		std::vector<size_t> vocabFrequencies;
		size_t iterated = 0;
		_ModelState globalState, tState;
		Dictionary dict;
		size_t realV = 0; // vocab size after removing stopwords
		size_t realN = 0; // total word size after removing stopwords
		size_t maxThreads[(size_t)ParallelScheme::size] = { 0, };

		std::unique_ptr<ThreadPool> cachedPool;

		void _saveModel(std::ostream& writer, bool fullModel) const
		{
			serializer::writeMany(writer, 
				serializer::MagicConstant{ _Derived::TMID },
				serializer::MagicConstant{ _Derived::TWID },
				dict, vocabFrequencies, realV);
			static_cast<const _Derived*>(this)->serializerWrite(writer);
			globalState.serializerWrite(writer);
			if (fullModel)
			{
				serializer::writeMany(writer, docs);
			}
			else
			{
				serializer::writeMany(writer, std::vector<size_t>{});
			}
		}

		void _loadModel(std::istream& reader)
		{
			serializer::readMany(reader, 
				serializer::MagicConstant{ _Derived::TMID },
				serializer::MagicConstant{ _Derived::TWID },
				dict, vocabFrequencies, realV);
			static_cast<_Derived*>(this)->serializerRead(reader);
			globalState.serializerRead(reader);
			serializer::readMany(reader, docs);
			realN = countRealN();
		}

		size_t _addDoc(const DocType& doc)
		{
			if (doc.words.empty()) return -1;
			size_t maxWid = *std::max_element(doc.words.begin(), doc.words.end());
			if (vocabFrequencies.size() <= maxWid) vocabFrequencies.resize(maxWid + 1);
			for (auto w : doc.words) ++vocabFrequencies[w];
			docs.emplace_back(doc);
			return docs.size() - 1;
		}

		size_t _addDoc(DocType&& doc)
		{
			if (doc.words.empty()) return -1;
			size_t maxWid = *std::max_element(doc.words.begin(), doc.words.end());
			if (vocabFrequencies.size() <= maxWid) vocabFrequencies.resize(maxWid + 1);
			for (auto w : doc.words) ++vocabFrequencies[w];
			docs.emplace_back(std::move(doc));
			return docs.size() - 1;
		}

		DocType _makeDoc(const std::vector<std::string>& words, FLOAT weight = 1)
		{
			DocType doc{ weight };
			//wOffsetByDoc.emplace_back(words.size());
			std::transform(words.begin(), words.end(), back_inserter(doc.words), [this](const std::string& w)
			{
				return dict.add(w);
			});
			return doc;
		}

		DocType _makeDocWithinVocab(const std::vector<std::string>& words, FLOAT weight = 1) const
		{
			DocType doc{ weight };
			//wOffsetByDoc.emplace_back(words.size());
			for (auto& w : words)
			{
				auto id = dict.toWid(w);
				if (id == (VID)-1) continue;
				doc.words.emplace_back(id);
			}
			return doc;
		}

		const DocType& _getDoc(size_t docId) const
		{
			return docs[docId];
		}

		void updateWeakArray()
		{
			wOffsetByDoc.emplace_back(0);
			for (auto& doc : docs)
			{
				wOffsetByDoc.emplace_back(wOffsetByDoc.back() + doc.words.size());
			}
			auto tx = [](_DocType& doc) { return &doc.words; };
			tvector<VID>::trade(words, 
				makeTransformIter(docs.begin(), tx),
				makeTransformIter(docs.end(), tx));
		}

		size_t countRealN() const
		{
			size_t n = 0;
			for (auto& doc : docs)
			{
				for (auto& w : doc.words)
				{
					if (w < realV) ++n;
				}
			}
			return n;
		}

		void removeStopwords(size_t minWordCnt, size_t removeTopN)
		{
			if (minWordCnt <= 1 && removeTopN == 0) realV = dict.size();
			std::vector<VID> order;
			sortAndWriteOrder(vocabFrequencies, order, removeTopN, std::greater<size_t>());
			realV = std::find_if(vocabFrequencies.begin(), vocabFrequencies.end() - std::min(removeTopN, vocabFrequencies.size()), [minWordCnt](size_t a) 
			{ 
				return a < minWordCnt; 
			}) - vocabFrequencies.begin();
			dict.reorder(order);
			realN = 0;
			for (auto& doc : docs)
			{
				for (auto& w : doc.words)
				{
					w = order[w];
					if (w < realV) ++realN;
				}
			}
		}

		int restoreFromTrainingError(const exception::TrainingError& e, ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			throw e;
		}

	public:
		TopicModel(const RandGen& _rg) : rg(_rg)
		{
		}

		size_t getNumDocs() const override
		{ 
			return docs.size(); 
		}

		size_t getN() const override
		{ 
			return realN; 
		}

		size_t getV() const override
		{
			return realV;
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t removeTopN = 0) override
		{
			maxThreads[(size_t)ParallelScheme::default_] = -1;
			maxThreads[(size_t)ParallelScheme::none] = -1;
			maxThreads[(size_t)ParallelScheme::copy_merge] = static_cast<_Derived*>(this)->template estimateMaxThreads<ParallelScheme::copy_merge>();
			maxThreads[(size_t)ParallelScheme::partition] = static_cast<_Derived*>(this)->template estimateMaxThreads<ParallelScheme::partition>();
		}

		static ParallelScheme getRealScheme(ParallelScheme ps)
		{
			switch (ps)
			{
			case ParallelScheme::default_:
				if ((_Flags & flags::partitioned_multisampling)) return ParallelScheme::partition;
				if ((_Flags & flags::shared_state)) return ParallelScheme::none;
				return ParallelScheme::copy_merge;
			case ParallelScheme::copy_merge:
				if ((_Flags & flags::shared_state)) THROW_ERROR_WITH_INFO(exception::InvalidArgument, 
					std::string{ "This model doesn't provide ParallelScheme::" } + toString(ps));
				break;
			case ParallelScheme::partition:
				if (!(_Flags & flags::partitioned_multisampling)) THROW_ERROR_WITH_INFO(exception::InvalidArgument,
					std::string{ "This model doesn't provide ParallelScheme::" } + toString(ps));
				break;
			}
			return ps;
		}

		int train(size_t iteration, size_t numWorkers, ParallelScheme ps) override
		{
			if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
			ps = getRealScheme(ps);
			numWorkers = std::min(numWorkers, maxThreads[(size_t)ps]);
			if (numWorkers == 1 || (_Flags & flags::shared_state)) ps = ParallelScheme::none;
			if (!cachedPool || cachedPool->getNumWorkers() != numWorkers)
			{
				cachedPool = make_unique<ThreadPool>(numWorkers);
			}

			std::vector<_ModelState> localData;
			std::vector<RandGen> localRG;
			for (size_t i = 0; i < numWorkers; ++i)
			{
				localRG.emplace_back(RandGen{rg()});
				if(ps == ParallelScheme::copy_merge) localData.emplace_back(static_cast<_Derived*>(this)->globalState);
			}

			if (ps == ParallelScheme::partition)
			{
				localData.resize(numWorkers);
				static_cast<_Derived*>(this)->updatePartition(*cachedPool, globalState, localData.data(), docs.begin(), docs.end(), 
					static_cast<_Derived*>(this)->eddTrain);
			}

			auto state = ps == ParallelScheme::none ? &globalState : localData.data();
			for (size_t i = 0; i < iteration; ++i)
			{
				while (1)
				{
					try
					{
						switch (ps)
						{
						case ParallelScheme::none:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::none>(
								*cachedPool, state, localRG.data());
							break;
						case ParallelScheme::copy_merge:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::copy_merge>(
								*cachedPool, state, localRG.data());
							break;
						case ParallelScheme::partition:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::partition>(
								*cachedPool, state, localRG.data());
							break;
						}
						break;
					}
					catch (const exception::TrainingError& e)
					{
						std::cerr << e.what() << std::endl;
						int ret = static_cast<_Derived*>(this)->restoreFromTrainingError(
							e, *cachedPool, state, localRG.data());
						if(ret < 0) return ret;
					}
				}
				++iterated;
			}
			return 0;
		}

		double getLLPerWord() const override
		{
			return words.empty() ? 0 : static_cast<const _Derived*>(this)->getLL() / realN;
		}

		double getPerplexity() const override
		{
			return exp(-getLLPerWord());
		}

		std::vector<FLOAT> getWidsByTopic(TID tid) const override
		{
			return static_cast<const _Derived*>(this)->_getWidsByTopic(tid);
		}

		std::vector<std::pair<VID, FLOAT>> getWidsByTopicSorted(TID tid, size_t topN) const
		{
			return extractTopN<VID>(static_cast<const _Derived*>(this)->_getWidsByTopic(tid), topN);
		}

		std::vector<std::pair<std::string, FLOAT>> vid2String(const std::vector<std::pair<VID, FLOAT>>& vids) const
		{
			std::vector<std::pair<std::string, FLOAT>> ret(vids.size());
			for (size_t i = 0; i < vids.size(); ++i)
			{
				ret[i] = std::make_pair(dict.toWord(vids[i].first), vids[i].second);
			}
			return ret;
		}

		std::vector<std::pair<std::string, FLOAT>> getWordsByTopicSorted(TID tid, size_t topN) const override
		{
			return vid2String(getWidsByTopicSorted(tid, topN));
		}

		std::vector<std::pair<VID, FLOAT>> getWidsByDocSorted(const DocumentBase* doc, size_t topN) const
		{
			std::vector<FLOAT> cnt(dict.size());
			for (auto w : doc->words) cnt[w] += 1;
			for (auto& c : cnt) c /= doc->words.size();
			return extractTopN<VID>(cnt, topN);
		}

		std::vector<std::pair<std::string, FLOAT>> getWordsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return vid2String(getWidsByDocSorted(doc, topN));
		}

		std::vector<double> infer(const std::vector<DocumentBase*>& docs, size_t maxIter, FLOAT tolerance, size_t numWorkers, ParallelScheme ps, bool together) const override
		{
			if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
			ps = getRealScheme(ps);
			if (numWorkers == 1) ps = ParallelScheme::none;
			auto tx = [](DocumentBase* p)->DocType& { return *static_cast<DocType*>(p); };
			auto b = makeTransformIter(docs.begin(), tx), e = makeTransformIter(docs.end(), tx);

			if (together)
			{
				switch (ps)
				{
				case ParallelScheme::none:
					return static_cast<const _Derived*>(this)->template _infer<true, ParallelScheme::none>(b, e, maxIter, tolerance, numWorkers);
				case ParallelScheme::copy_merge:
					return static_cast<const _Derived*>(this)->template _infer<true, ParallelScheme::copy_merge>(b, e, maxIter, tolerance, numWorkers);
				case ParallelScheme::partition:
					return static_cast<const _Derived*>(this)->template _infer<true, ParallelScheme::partition>(b, e, maxIter, tolerance, numWorkers);
				}
			}
			else
			{
				switch (ps)
				{
				case ParallelScheme::none:
					return static_cast<const _Derived*>(this)->template _infer<false, ParallelScheme::none>(b, e, maxIter, tolerance, numWorkers);
				case ParallelScheme::copy_merge:
					return static_cast<const _Derived*>(this)->template _infer<false, ParallelScheme::copy_merge>(b, e, maxIter, tolerance, numWorkers);
				case ParallelScheme::partition:
					return static_cast<const _Derived*>(this)->template _infer<false, ParallelScheme::partition>(b, e, maxIter, tolerance, numWorkers);
				}
			}
			THROW_ERROR_WITH_INFO(exception::InvalidArgument, "invalid ParallelScheme");
		}

		std::vector<FLOAT> getTopicsByDoc(const DocumentBase* doc) const override
		{
			return static_cast<const _Derived*>(this)->getTopicsByDoc(*static_cast<const DocType*>(doc));
		}

		std::vector<std::pair<TID, FLOAT>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return extractTopN<TID>(getTopicsByDoc(doc), topN);
		}


		const DocumentBase* getDoc(size_t docId) const override
		{
			return &_getDoc(docId);
		}

		const Dictionary& getVocabDict() const override
		{
			return dict;
		}

		const std::vector<size_t>& getVocabFrequencies() const override
		{
			return vocabFrequencies;
		}

		void saveModel(std::ostream& writer, bool fullModel) const override
		{ 
			static_cast<const _Derived*>(this)->_saveModel(writer, fullModel);
		}

		void loadModel(std::istream& reader) override
		{ 
			static_cast<_Derived*>(this)->_loadModel(reader);
			static_cast<_Derived*>(this)->prepare(false);
		}
	};

}
