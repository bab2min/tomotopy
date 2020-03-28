#pragma once
#include <unordered_set>
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
		Float weight = 1;
		tvector<Vid> words; // word id of each word
		std::vector<uint32_t> wOrder; // original word order (optional)

		std::string rawStr;
		std::vector<uint32_t> origWordPos;
		std::vector<uint16_t> origWordLen;
		DocumentBase(Float _weight = 1) : weight(_weight) {}
		virtual ~DocumentBase() {}

		DEFINE_SERIALIZER_WITH_VERSION(0, serializer::to_key("Docu"), weight, words, wOrder);
		DEFINE_TAGGED_SERIALIZER_WITH_VERSION(1, 0x00010001, weight, words, wOrder, rawStr, origWordPos, origWordLen);
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

	class RawDocTokenizer
	{
	public:
		using Token = std::tuple<std::string, uint32_t, uint32_t, bool>;
		using Factory = std::function<RawDocTokenizer(const std::string&)>;
	private:
		std::function<Token()> fnNext;
	public:
		class Iterator
		{
			RawDocTokenizer* p = nullptr;
			bool end = true;
			std::tuple<std::string, uint32_t, uint32_t> value;
		public:
			Iterator()
			{
			}

			Iterator(RawDocTokenizer* _p)
				: p{ _p }, end{ false }
			{
				operator++();
			}

			std::tuple<std::string, uint32_t, uint32_t>& operator*()
			{
				return value;
			}

			Iterator& operator++()
			{
				auto v = p->fnNext();
				if (std::get<3>(v))
				{
					end = true;
				}
				else
				{
					value = std::make_tuple(std::get<0>(v), std::get<1>(v), std::get<2>(v));
				}
				return *this;
			}

			bool operator==(const Iterator& o) const
			{
				return o.end && end;
			}

			bool operator!=(const Iterator& o) const
			{
				return !operator==(o);
			}
		};

		template<typename _Fn>
		RawDocTokenizer(_Fn&& fn) : fnNext{ std::forward<_Fn>(fn) }
		{
		}

		Iterator begin()
		{
			return Iterator{ this };
		}

		Iterator end()
		{
			return Iterator{};
		}
	};

	class ITopicModel
	{
	public:
		virtual void saveModel(std::ostream& writer, bool fullModel) const = 0;
		virtual void loadModel(std::istream& reader) = 0;
		virtual const DocumentBase* getDoc(size_t docId) const = 0;

		virtual void updateVocab(const std::vector<std::string>& words) = 0;

		virtual double getLLPerWord() const = 0;
		virtual double getPerplexity() const = 0;
		virtual size_t getV() const = 0;
		virtual size_t getN() const = 0;
		virtual size_t getNumDocs() const = 0;
		virtual const Dictionary& getVocabDict() const = 0;
		virtual const std::vector<size_t>& getVocabCf() const = 0;
		virtual const std::vector<size_t>& getVocabDf() const = 0;

		virtual int train(size_t iteration, size_t numWorkers, ParallelScheme ps = ParallelScheme::default_) = 0;
		virtual void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0) = 0;
		
		virtual size_t getK() const = 0;
		virtual std::vector<Float> getWidsByTopic(Tid tid) const = 0;
		virtual std::vector<std::pair<std::string, Float>> getWordsByTopicSorted(Tid tid, size_t topN) const = 0;

		virtual std::vector<std::pair<std::string, Float>> getWordsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		
		virtual std::vector<Float> getTopicsByDoc(const DocumentBase* doc) const = 0;
		virtual std::vector<std::pair<Tid, Float>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		virtual std::vector<double> infer(const std::vector<DocumentBase*>& docs, size_t maxIter, Float tolerance, size_t numWorkers, ParallelScheme ps, bool together) const = 0;
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
		std::vector<Vid> words;
		std::vector<uint32_t> wOffsetByDoc;

		std::vector<DocType> docs;
		std::vector<size_t> vocabCf;
		std::vector<size_t> vocabDf;
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
				serializer::to_keyz(static_cast<const _Derived*>(this)->TMID),
				serializer::to_keyz(static_cast<const _Derived*>(this)->TWID));
			serializer::writeTaggedMany(writer, 0x00010001,
				serializer::to_keyz("dict"), dict, 
				serializer::to_keyz("vocabCf"), vocabCf,
				serializer::to_keyz("vocabDf"), vocabDf,
				serializer::to_keyz("realV"), realV);
			serializer::writeMany(writer, *static_cast<const _Derived*>(this));
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
			auto start_pos = reader.tellg();
			try
			{
				serializer::readMany(reader, 
					serializer::to_keyz(static_cast<_Derived*>(this)->TMID),
					serializer::to_keyz(static_cast<_Derived*>(this)->TWID));
				serializer::readTaggedMany(reader, 0x00010001, 
					serializer::to_keyz("dict"), dict,
					serializer::to_keyz("vocabCf"), vocabCf,
					serializer::to_keyz("vocabDf"), vocabDf,
					serializer::to_keyz("realV"), realV);
			}
			catch (const std::ios_base::failure&)
			{
				reader.seekg(start_pos);
				serializer::readMany(reader,
					serializer::to_key(static_cast<_Derived*>(this)->TMID),
					serializer::to_key(static_cast<_Derived*>(this)->TWID),
					dict, vocabCf, realV);
			}
			serializer::readMany(reader, *static_cast<_Derived*>(this));
			globalState.serializerRead(reader);
			serializer::readMany(reader, docs);
			realN = countRealN();
		}

		template<typename _DocTy>
		typename std::enable_if<std::is_same<DocType, 
			typename std::remove_reference<typename std::remove_cv<_DocTy>::type>::type
		>::value, size_t>::type _addDoc(_DocTy&& doc)
		{
			if (doc.words.empty()) return -1;
			size_t maxWid = *std::max_element(doc.words.begin(), doc.words.end());
			if (vocabCf.size() <= maxWid)
			{
				vocabCf.resize(maxWid + 1);
				vocabDf.resize(maxWid + 1);
			}
			for (auto w : doc.words) ++vocabCf[w];
			std::unordered_set<Vid> uniq{ doc.words.begin(), doc.words.end() };
			for (auto w : uniq) ++vocabDf[w];
			docs.emplace_back(std::forward<_DocTy>(doc));
			return docs.size() - 1;
		}

		template<bool _const = false>
		DocType _makeDoc(const std::vector<std::string>& words, Float weight = 1)
		{
			DocType doc{ weight };
			for (auto& w : words)
			{
				Vid id;
				if (_const)
				{
					id = dict.toWid(w);
					if (id == (Vid)-1) continue;
				}
				else
				{
					id = dict.add(w);
				}
				doc.words.emplace_back(id);
			}
			return doc;
		}

		DocType _makeRawDoc(const std::string& rawStr, const std::vector<Vid>& words, 
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len, Float weight = 1) const
		{
			DocType doc{ weight };
			doc.rawStr = rawStr;
			for (auto& w : words) doc.words.emplace_back(w);
			doc.origWordPos = pos;
			doc.origWordLen = len;
			return doc;
		}

		template<bool _const, typename _FnTokenizer>
		DocType _makeRawDoc(const std::string& rawStr, _FnTokenizer&& tokenizer, Float weight = 1)
		{
			DocType doc{ weight };
			doc.rawStr = rawStr;
			for (auto& p : tokenizer(doc.rawStr))
			{
				Vid wid;
				if (_const)
				{
					wid = dict.toWid(std::get<0>(p));
					if (wid == (Vid)-1) continue;
				}
				else
				{
					wid = dict.add(std::get<0>(p));
				}
				auto pos = std::get<1>(p);
				auto len = std::get<2>(p);
				doc.words.emplace_back(wid);
				doc.origWordPos.emplace_back(pos);
				doc.origWordLen.emplace_back(len);
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
			tvector<Vid>::trade(words, 
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

		void removeStopwords(size_t minWordCnt, size_t minWordDf, size_t removeTopN)
		{
			if (minWordCnt <= 1 && minWordDf <= 1 && removeTopN == 0) realV = dict.size();
			std::vector<std::pair<size_t, size_t>> vocabCfDf;
			for (size_t i = 0; i < vocabCf.size(); ++i)
			{
				vocabCfDf.emplace_back(vocabCf[i], vocabDf[i]);
			}

			std::vector<Vid> order;
			sortAndWriteOrder(vocabCfDf, order, removeTopN, [&](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b)
			{
				if (a.first < minWordCnt || a.second < minWordDf)
				{
					if (b.first < minWordCnt || b.second < minWordDf)
					{
						return a > b;
					}
					return false;
				}
				if (b.first < minWordCnt || b.second < minWordDf)
				{
					return true;
				}
				return a > b;
			});
			realV = std::find_if(vocabCfDf.begin(), vocabCfDf.end() - std::min(removeTopN, vocabCfDf.size()), [&](const std::pair<size_t, size_t>& a)
			{ 
				return a.first < minWordCnt || a.second < minWordDf;
			}) - vocabCfDf.begin();

			for (size_t i = 0; i < vocabCfDf.size(); ++i)
			{
				vocabCf[i] = vocabCfDf[i].first;
				vocabDf[i] = vocabCfDf[i].second;
			}

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

		void updateVocab(const std::vector<std::string>& words) override
		{
			if(dict.size()) THROW_ERROR_WITH_INFO(exception::InvalidArgument, "updateVocab after addDoc");
			for(auto& w : words) dict.add(w);
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0) override
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

		size_t getK() const override
		{
			return 0;
		}

		std::vector<Float> getWidsByTopic(Tid tid) const override
		{
			return static_cast<const _Derived*>(this)->_getWidsByTopic(tid);
		}

		std::vector<std::pair<Vid, Float>> getWidsByTopicSorted(Tid tid, size_t topN) const
		{
			return extractTopN<Vid>(static_cast<const _Derived*>(this)->_getWidsByTopic(tid), topN);
		}

		std::vector<std::pair<std::string, Float>> vid2String(const std::vector<std::pair<Vid, Float>>& vids) const
		{
			std::vector<std::pair<std::string, Float>> ret(vids.size());
			for (size_t i = 0; i < vids.size(); ++i)
			{
				ret[i] = std::make_pair(dict.toWord(vids[i].first), vids[i].second);
			}
			return ret;
		}

		std::vector<std::pair<std::string, Float>> getWordsByTopicSorted(Tid tid, size_t topN) const override
		{
			return vid2String(getWidsByTopicSorted(tid, topN));
		}

		std::vector<std::pair<Vid, Float>> getWidsByDocSorted(const DocumentBase* doc, size_t topN) const
		{
			std::vector<Float> cnt(dict.size());
			for (auto w : doc->words) cnt[w] += 1;
			for (auto& c : cnt) c /= doc->words.size();
			return extractTopN<Vid>(cnt, topN);
		}

		std::vector<std::pair<std::string, Float>> getWordsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return vid2String(getWidsByDocSorted(doc, topN));
		}

		std::vector<double> infer(const std::vector<DocumentBase*>& docs, size_t maxIter, Float tolerance, size_t numWorkers, ParallelScheme ps, bool together) const override
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

		std::vector<Float> getTopicsByDoc(const DocumentBase* doc) const override
		{
			return static_cast<const _Derived*>(this)->getTopicsByDoc(*static_cast<const DocType*>(doc));
		}

		std::vector<std::pair<Tid, Float>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return extractTopN<Tid>(getTopicsByDoc(doc), topN);
		}


		const DocumentBase* getDoc(size_t docId) const override
		{
			return &_getDoc(docId);
		}

		const Dictionary& getVocabDict() const override
		{
			return dict;
		}

		const std::vector<size_t>& getVocabCf() const override
		{
			return vocabCf;
		}

		const std::vector<size_t>& getVocabDf() const override
		{
			return vocabDf;
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
