#pragma once
#include <numeric>
#include <unordered_set>
#include "../Utils/Utils.hpp"
#include "../Utils/Dictionary.h"
#include "../Utils/tvector.hpp"
#include "../Utils/ThreadPool.hpp"
#include "../Utils/serializer.hpp"
#include "../Utils/exception.h"
#include "../Utils/SharedString.hpp"
#include <EigenRand/EigenRand>
#include <mapbox/variant.hpp>

namespace tomoto
{
	using RandGen = Eigen::Rand::P8_mt19937_64_32;
	using ScalarRandGen = Eigen::Rand::UniversalRandomEngine<uint32_t, std::mt19937_64>;

	using Vector = Eigen::Matrix<Float, -1, 1>;
	using Matrix = Eigen::Matrix<Float, -1, -1>;

	struct RawDocKernel
	{
		Float weight = 1;
		SharedString docUid;
		SharedString rawStr;
		std::vector<uint32_t> origWordPos;
		std::vector<uint16_t> origWordLen;

		RawDocKernel(const RawDocKernel&) = default;
		RawDocKernel(RawDocKernel&&) = default;

		RawDocKernel(Float _weight = 1, const SharedString& _docUid = {})
			: weight{ _weight }, docUid{ _docUid }
		{
		}
	};

	struct RawDoc : public RawDocKernel
	{
		using Var = mapbox::util::variant<
			std::string, uint32_t, Float,
			std::vector<std::string>, std::vector<uint32_t>, std::vector<Float>,
			std::shared_ptr<void>
		>;
		using MiscType = std::unordered_map<std::string, Var>;

		std::vector<Vid> words;
		std::vector<std::string> rawWords;
		MiscType misc;

		RawDoc() = default;
		RawDoc(const RawDoc&) = default;
		RawDoc(RawDoc&&) = default;

		RawDoc(const RawDocKernel& o) 
			: RawDocKernel{ o }
		{
		}

		template<typename _Ty>
		const _Ty& getMisc(const std::string& name) const
		{
			auto it = misc.find(name);
			if (it == misc.end()) throw exc::InvalidArgument{ "There is no value named `" + name + "` in misc data" };
			if (!it->second.template is<_Ty>()) throw exc::InvalidArgument{ "Value named `" + name + "` is not in right type." };
			return it->second.template get<_Ty>();
		}

		template<typename _Ty>
		_Ty getMiscDefault(const std::string& name) const
		{
			auto it = misc.find(name);
			if (it == misc.end()) return {};
			if (!it->second.template is<_Ty>()) throw exc::InvalidArgument{ "Value named `" + name + "` is not in right type." };
			return it->second.template get<_Ty>();
		}
	};

	class ITopicModel;

	class DocumentBase : public RawDocKernel
	{
	public:
		tvector<Vid> words; // word id of each word
		std::vector<uint32_t> wOrder; // original word order (optional)
		
		DocumentBase(const DocumentBase&) = default;
		DocumentBase(DocumentBase&&) = default;

		DocumentBase(const RawDocKernel& o) 
			: RawDocKernel{ o }
		{
		}

		DocumentBase(Float _weight = 1, const SharedString& _docUid = {})
			: RawDocKernel{ _weight, _docUid }
		{
		}

		virtual ~DocumentBase() {}

		virtual RawDoc::MiscType makeMisc(const ITopicModel*) const
		{
			return {};
		}

		virtual operator RawDoc() const
		{
			RawDoc raw{ *this };
			if (wOrder.empty())
			{
				raw.words.insert(raw.words.begin(), words.begin(), words.end());
			}
			else
			{
				raw.words.resize(words.size());
				for (size_t i = 0; i < words.size(); ++i)
				{
					raw.words[i] = words[wOrder[i]];
				}
			}
			//raw.misc = makeMisc();
			return raw;
		}

		DEFINE_SERIALIZER_WITH_VERSION(0, serializer::to_key("Docu"), weight, words, wOrder);
		DEFINE_TAGGED_SERIALIZER_WITH_VERSION(1, 0x00010001, weight, words, wOrder, 
			rawStr, origWordPos, origWordLen,
			docUid
		);
	};

	enum class ParallelScheme { default_, none, copy_merge, partition, size };
	enum class GlobalSampler { train, freeze_topics, inference, size };

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
		virtual void saveModel(std::ostream& writer, bool fullModel, 
			const std::vector<uint8_t>* extra_data = nullptr) const = 0;
		virtual void loadModel(std::istream& reader, 
			std::vector<uint8_t>* extra_data = nullptr) = 0;

		virtual std::unique_ptr<ITopicModel> copy() const = 0;

		virtual const DocumentBase* getDoc(size_t docId) const = 0;
		virtual size_t getDocIdByUid(const std::string& docUid) const = 0;

		// it tokenizes rawDoc.rawStr to get words, pos and len of the document
		virtual size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const = 0;

		// it uses words, pos and len of rawDoc itself.
		virtual size_t addDoc(const RawDoc& rawDoc) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const = 0;

		virtual bool updateVocab(const std::vector<std::string>& words) = 0;

		virtual double getDocLL(const DocumentBase* doc) const = 0;
		virtual double getStateLL() const = 0;

		virtual double getLLPerWord() const = 0;
		virtual double getPerplexity() const = 0;
		virtual uint64_t getV() const = 0;
		virtual uint64_t getN() const = 0;
		virtual size_t getNumDocs() const = 0;
		virtual const Dictionary& getVocabDict() const = 0;
		virtual const std::vector<uint64_t>& getVocabCf() const = 0;
		virtual std::vector<double> getVocabWeightedCf() const = 0;
		virtual const std::vector<uint64_t>& getVocabDf() const = 0;

		virtual int train(size_t iteration, size_t numWorkers, ParallelScheme ps = ParallelScheme::default_, bool freeze_topics = false) = 0;
		virtual size_t getGlobalStep() const = 0;
		virtual void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0, bool updateStopwords = true) = 0;
		
		virtual size_t getK() const = 0;
		virtual size_t getNumTopicsForPrior() const = 0;
		virtual std::vector<Float> getWidsByTopic(size_t tid, bool normalize = true) const = 0;
		virtual std::vector<std::pair<std::string, Float>> getWordsByTopicSorted(size_t tid, size_t topN) const = 0;

		virtual std::vector<std::pair<std::string, Float>> getWordsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		
		virtual std::vector<Float> getTopicsByDoc(const DocumentBase* doc, bool normalize = true) const = 0;
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

	template<typename _RandGen, size_t _Flags, typename _Interface, typename _Derived,
		typename _DocType, typename _ModelState
	>
	class TopicModel : public _Interface
	{
		friend class Document;
	public:
		using DocType = _DocType;
	protected:
		_RandGen rg;
		std::vector<_RandGen> localRG;
		std::vector<Vid> words;
		std::vector<uint32_t> wOffsetByDoc;

		std::vector<DocType> docs;
		std::vector<uint64_t> vocabCf;
		std::vector<uint64_t> vocabDf;
		std::unordered_map<SharedString, size_t> uidMap;
		size_t globalStep = 0;
		_ModelState globalState, tState;
		Dictionary dict;
		uint64_t realV = 0; // vocab size after removing stopwords
		uint64_t realN = 0; // total word size after removing stopwords
		double weightedN = 0;
		size_t maxThreads[(size_t)ParallelScheme::size] = { 0, };
		size_t minWordCf = 0, minWordDf = 0, removeTopN = 0;

		PreventCopy<std::unique_ptr<ThreadPool>> cachedPool;

		void _saveModel(std::ostream& writer, bool fullModel, const std::vector<uint8_t>* extra_data) const
		{
			serializer::writeMany(writer,
				serializer::to_keyz(static_cast<const _Derived*>(this)->tmid()),
				serializer::to_keyz(static_cast<const _Derived*>(this)->twid())
			);
			serializer::writeTaggedMany(writer, 0x00010001,
				serializer::to_keyz("dict"), dict, 
				serializer::to_keyz("vocabCf"), vocabCf,
				serializer::to_keyz("vocabDf"), vocabDf,
				serializer::to_keyz("realV"), realV,
				serializer::to_keyz("globalStep"), globalStep,
				serializer::to_keyz("extra"), extra_data ? *extra_data : std::vector<uint8_t>(0)
			);
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

		void _loadModel(std::istream& reader, std::vector<uint8_t>* extra_data)
		{
			auto start_pos = reader.tellg();
			try
			{
				std::vector<uint8_t> extra;
				serializer::readMany(reader, 
					serializer::to_keyz(static_cast<_Derived*>(this)->tmid()),
					serializer::to_keyz(static_cast<_Derived*>(this)->twid())
				);
				serializer::readTaggedMany(reader, 0x00010001, 
					serializer::to_keyz("dict"), dict,
					serializer::to_keyz("vocabCf"), vocabCf,
					serializer::to_keyz("vocabDf"), vocabDf,
					serializer::to_keyz("realV"), realV,
					serializer::to_keyz("globalStep"), globalStep,
					serializer::to_keyz("extra"), extra);
				if (extra_data) *extra_data = std::move(extra);
			}
			catch (const std::ios_base::failure&)
			{
				reader.seekg(start_pos);
				serializer::readMany(reader,
					serializer::to_key(static_cast<_Derived*>(this)->tmid()),
					serializer::to_key(static_cast<_Derived*>(this)->twid()),
					dict, vocabCf, realV
				);
			}
			serializer::readMany(reader, *static_cast<_Derived*>(this));
			globalState.serializerRead(reader);
			serializer::readMany(reader, docs);
			auto p = countRealN();
			realN = p.first;
			weightedN = p.second;
		}

		template<typename _DocTy>
		typename std::enable_if<std::is_same<DocType, 
			typename std::remove_reference<typename std::remove_cv<_DocTy>::type>::type
		>::value, size_t>::type _addDoc(_DocTy&& doc)
		{
			if (doc.words.empty()) return -1;
			if (!doc.docUid.empty() && uidMap.count(doc.docUid))
				throw exc::InvalidArgument{ "there is a document with uid = '" + std::string{ doc.docUid } + "' already." };
			size_t maxWid = *std::max_element(doc.words.begin(), doc.words.end());
			if (vocabCf.size() <= maxWid)
			{
				vocabCf.resize(maxWid + 1);
				vocabDf.resize(maxWid + 1);
			}
			for (auto w : doc.words) ++vocabCf[w];
			std::unordered_set<Vid> uniq{ doc.words.begin(), doc.words.end() };
			for (auto w : uniq) ++vocabDf[w];
			if (!doc.docUid.empty()) uidMap.emplace(doc.docUid, docs.size());
			docs.emplace_back(std::forward<_DocTy>(doc));
			return docs.size() - 1;
		}

		template<bool _const = false>
		DocType _makeFromRawDoc(const RawDoc& rawDoc)
		{
			DocType doc{ rawDoc };
			if (!rawDoc.rawWords.empty())
			{
				for (auto& w : rawDoc.rawWords)
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
			}
			else if(!rawDoc.words.empty())
			{
				for (auto& w : rawDoc.words) doc.words.emplace_back(w);
			}
			else
			{
				throw exc::EmptyWordArgument{ "Either `words` or `rawWords` must be filled." };
			}
			return doc;
		}

		template<bool _const, typename _FnTokenizer>
		DocType _makeFromRawDoc(const RawDoc& rawDoc, _FnTokenizer&& tokenizer)
		{
			DocType doc{ rawDoc };
			doc.rawStr = rawDoc.rawStr;
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
				makeTransformIter(docs.end(), tx)
			);
		}

		void updateForCopy()
		{
			size_t offset = 0;
			for (auto& doc : docs)
			{
				size_t size = doc.words.size();
				doc.words = tvector<Vid>{ words.data() + offset, size };
				offset += size;
			}
		}

		std::pair<size_t, double> countRealN() const
		{
			size_t n = 0;
			double weighted = 0;
			for (auto& doc : docs)
			{
				for (size_t i = 0; i < doc.words.size(); ++i)
				{
					auto w = doc.words[i];
					if (w < realV)
					{
						++n;
						weighted += doc.wordWeights.empty() ? 1 : doc.wordWeights[i];
					}
				}
			}
			return std::make_pair(n, weighted);
		}

		void removeStopwords(size_t minWordCnt, size_t minWordDf, size_t removeTopN)
		{
			if (minWordCnt <= 1 && minWordDf <= 1 && removeTopN == 0) realV = dict.size();
			this->minWordCf = minWordCnt;
			this->minWordDf = minWordDf;
			this->removeTopN = removeTopN;
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
			for (auto& doc : docs)
			{
				for (auto& w : doc.words) w = order[w];
			}
		}

		int restoreFromTrainingError(const exc::TrainingError& e, ThreadPool& pool, _ModelState* localData, _RandGen* rgs)
		{
			throw e;
		}

	public:
		TopicModel(size_t _rg) : rg(_rg)
		{
		}

		TopicModel(const TopicModel&) = default;

		std::unique_ptr<ITopicModel> copy() const override
		{
			auto ret = std::make_unique<_Derived>(*static_cast<const _Derived*>(this));
			ret->updateForCopy();
			return ret;
		}

		size_t getNumDocs() const override
		{ 
			return docs.size(); 
		}

		uint64_t getN() const override
		{ 
			return realN; 
		}

		uint64_t getV() const override
		{
			return realV;
		}

		bool updateVocab(const std::vector<std::string>& words) override
		{
			bool empty = dict.size() == 0;
			for (auto& w : words) dict.add(w);
			return empty;
		}

		void prepare(bool initDocs = true, size_t minWordCnt = 0, size_t minWordDf = 0, size_t removeTopN = 0, bool updateStopwords = true) override
		{
			auto p = countRealN();
			realN = p.first;
			weightedN = p.second;

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
				if ((_Flags & flags::shared_state)) THROW_ERROR_WITH_INFO(exc::InvalidArgument, 
					std::string{ "This model doesn't provide ParallelScheme::" } + toString(ps));
				break;
			case ParallelScheme::partition:
				if (!(_Flags & flags::partitioned_multisampling)) THROW_ERROR_WITH_INFO(exc::InvalidArgument,
					std::string{ "This model doesn't provide ParallelScheme::" } + toString(ps));
				break;
			}
			return ps;
		}

		int train(size_t iteration, size_t numWorkers, ParallelScheme ps, bool freeze_topics = false) override
		{
			if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
			ps = getRealScheme(ps);
			numWorkers = std::min(numWorkers, maxThreads[(size_t)ps]);
			if (numWorkers == 1 || (_Flags & flags::shared_state)) ps = ParallelScheme::none;
			if (!cachedPool || cachedPool->getNumWorkers() != numWorkers)
			{
				cachedPool = std::make_unique<ThreadPool>(numWorkers);
			}

			std::vector<_ModelState> localData;

			while(localRG.size() < numWorkers)
			{
				localRG.emplace_back(rg());
			}

			if (ps == ParallelScheme::copy_merge)
			{
				for (size_t i = 0; i < numWorkers; ++i)
				{
					localData.emplace_back(static_cast<_Derived*>(this)->globalState);
				}
			}
			else if (ps == ParallelScheme::partition)
			{
				localData.resize(numWorkers);
				static_cast<_Derived*>(this)->updatePartition(
					*cachedPool, globalState, localData.data(), docs.begin(), docs.end(), 
					static_cast<_Derived*>(this)->eddTrain
				);
			}

			auto state = ps == ParallelScheme::none ? &globalState : localData.data();
			for (size_t i = 0; i < iteration; ++i)
			{
				size_t retry;
				for (retry = 0; retry < 10; ++retry)
				{
					try
					{
						switch (ps)
						{
						case ParallelScheme::none:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::none>(
								*cachedPool, state, localRG.data(), freeze_topics);
							break;
						case ParallelScheme::copy_merge:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::copy_merge>(
								*cachedPool, state, localRG.data(), freeze_topics);
							break;
						case ParallelScheme::partition:
							static_cast<_Derived*>(this)->template trainOne<ParallelScheme::partition>(
								*cachedPool, state, localRG.data(), freeze_topics);
							break;
						}
						break;
					}
					catch (const exc::TrainingError& e)
					{
						std::cerr << e.what() << std::endl;
						int ret = static_cast<_Derived*>(this)->restoreFromTrainingError(
							e, *cachedPool, state, localRG.data());
						if(ret < 0) return ret;
					}
				}
				if (retry >= 10) return -1;
				++globalStep;
			}
			return 0;
		}

		double getLLPerWord() const override
		{
			return words.empty() ? 0 : static_cast<const _Derived*>(this)->getLL() / weightedN;
		}

		double getPerplexity() const override
		{
			return exp(-getLLPerWord());
		}

		size_t getK() const override
		{
			return 0;
		}

		size_t getNumTopicsForPrior() const override
		{
			return this->getK();
		}

		std::vector<Float> getWidsByTopic(size_t tid, bool normalize) const override
		{
			return static_cast<const _Derived*>(this)->_getWidsByTopic(tid, normalize);
		}

		std::vector<std::pair<Vid, Float>> getWidsByTopicSorted(size_t tid, size_t topN) const
		{
			return extractTopN<Vid>(static_cast<const _Derived*>(this)->_getWidsByTopic(tid, true), topN);
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

		std::vector<std::pair<std::string, Float>> getWordsByTopicSorted(size_t tid, size_t topN) const override
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

		double getDocLL(const DocumentBase* doc) const override
		{
			auto* p = dynamic_cast<const DocType*>(doc);
			if (!p) throw exc::InvalidArgument{ "wrong `doc` type." };
			return static_cast<const _Derived*>(this)->getLLDocs(p, p + 1);
		}

		double getStateLL() const override
		{
			return static_cast<const _Derived*>(this)->getLLRest(this->globalState);
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
			THROW_ERROR_WITH_INFO(exc::InvalidArgument, "invalid ParallelScheme");
		}

		std::vector<Float> getTopicsByDoc(const DocumentBase* doc, bool normalize) const override
		{
			return static_cast<const _Derived*>(this)->_getTopicsByDoc(*static_cast<const DocType*>(doc), normalize);
		}

		std::vector<std::pair<Tid, Float>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const override
		{
			return extractTopN<Tid>(getTopicsByDoc(doc, true), topN);
		}

		const DocumentBase* getDoc(size_t docId) const override
		{
			return &_getDoc(docId);
		}

		size_t getDocIdByUid(const std::string& docUid) const override
		{
			auto it = uidMap.find(SharedString{ docUid });
			if (it == uidMap.end()) return -1;
			return it->second;
		}

		size_t getGlobalStep() const override
		{
			return globalStep;
		}

		const Dictionary& getVocabDict() const override
		{
			return dict;
		}

		const std::vector<uint64_t>& getVocabCf() const override
		{
			return vocabCf;
		}

		std::vector<double> getVocabWeightedCf() const override
		{
			std::vector<double> ret(realV);
			for (auto& doc : docs)
			{
				for (size_t i = 0; i < doc.words.size(); ++i)
				{
					if (doc.words[i] >= realV) continue;
					ret[doc.words[i]] += doc.wordWeights.empty() ? 1 : doc.wordWeights[i];
				}
			}
			return ret;
		}

		const std::vector<uint64_t>& getVocabDf() const override
		{
			return vocabDf;
		}

		void saveModel(std::ostream& writer, bool fullModel, const std::vector<uint8_t>* extra_data) const override
		{ 
			static_cast<const _Derived*>(this)->_saveModel(writer, fullModel, extra_data);
		}

		void loadModel(std::istream& reader, std::vector<uint8_t>* extra_data) override
		{ 
			static_cast<_Derived*>(this)->_loadModel(reader, extra_data);
			static_cast<_Derived*>(this)->prepare(false);
		}
	};

}
