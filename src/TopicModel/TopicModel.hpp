#pragma once
#include "../Utils/Dictionary.h"
#include "../Utils/tvector.hpp"
#include "../Utils/ThreadPool.hpp"
#include "../Utils/serializer.hpp"

namespace tomoto
{
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

		virtual void train(size_t iteration, size_t numWorkers) = 0;
		virtual void prepare(bool initDocs = true) = 0;
		virtual std::vector<FLOAT> getWidsByTopic(TID tid) const = 0;
		virtual std::vector<std::pair<std::string, FLOAT>> getWordsByTopicSorted(TID tid, size_t topN) const = 0;
		
		virtual std::vector<FLOAT> getTopicsByDoc(const DocumentBase* doc) const = 0;
		virtual std::vector<std::pair<TID, FLOAT>> getTopicsByDocSorted(const DocumentBase* doc, size_t topN) const = 0;
		virtual std::vector<FLOAT> infer(DocumentBase* doc, size_t maxIter, FLOAT tolerance, FLOAT* ll = nullptr) const = 0;
		virtual FLOAT infer(const std::vector<DocumentBase*>& docs, size_t maxIter, FLOAT tolerance) const = 0;
		virtual ~ITopicModel() {}
	};

	template<class _TyKey, class _TyValue>
	static std::vector<std::pair<_TyKey, _TyValue>> extractTopN(const std::vector<_TyValue>& vec, size_t topN)
	{
		std::vector<std::pair<_TyKey, _TyValue>> ret;
		_TyKey k = 0;
		for (auto&& t : vec)
		{
			ret.emplace_back(std::make_pair(k++, t));
		}
		std::sort(ret.begin(), ret.end(), [](const auto& a, const auto& b)
		{
			return a.second > b.second;
		});
		if (topN < ret.size()) ret.erase(ret.begin() + topN, ret.end());
		return ret;
	}

	template<typename _Interface, typename _Derived, typename _DocType, typename _ModelState>
	class TopicModel : public _Interface
	{
		friend class Document;
	public:
		using DocType = _DocType;
	protected:
		RANDGEN rg;
		std::vector<VID> words;
		std::vector<uint32_t> wOffsetByDoc;

		std::vector<DocType> docs;
		size_t iterated = 0;
		_ModelState globalState, tState;
		Dictionary dict;

		void _saveModel(std::ostream& writer, bool fullModel) const
		{
			serializer::writeMany(writer, 
				serializer::MagicConstant{ _Derived::TMID },
				serializer::MagicConstant{ _Derived::TWID },
				dict);
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
				dict);
			static_cast<_Derived*>(this)->serializerRead(reader);
			globalState.serializerRead(reader);
			serializer::readMany(reader, docs);
		}

		size_t _addDoc(const DocType& doc)
		{
			if(!doc.words.empty()) docs.emplace_back(doc);
			return docs.size() - 1;
		}

		size_t _addDoc(DocType&& doc)
		{
			if (!doc.words.empty()) docs.emplace_back(std::move(doc));
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
			std::vector<tvector<VID>*> srcs;
			srcs.reserve(docs.size());
			wOffsetByDoc.emplace_back(0);
			for (auto&& doc : docs)
			{
				srcs.emplace_back(&doc.words);
				wOffsetByDoc.emplace_back(wOffsetByDoc.back() + doc.words.size());
			}
			tvector<VID>::trade(words, srcs.begin(), srcs.end());
		}

	public:
		TopicModel(const RANDGEN& _rg) : rg(_rg)
		{
		}

		size_t getNumDocs() const override
		{ 
			return docs.size(); 
		}

		size_t getN() const override
		{ 
			return words.size(); 
		}

		size_t getV() const override
		{
			return dict.size();
		}

		void prepare(bool initDocs = true) override
		{
		}

		void train(size_t iteration, size_t numWorkers) override
		{
			if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
			ThreadPool pool(numWorkers);
			std::vector<_ModelState> localData;
			std::vector<RANDGEN> localRG;
			for (size_t i = 0; i < numWorkers; ++i)
			{
				localRG.emplace_back(RANDGEN{rg()});
				localData.emplace_back(static_cast<_Derived*>(this)->globalState);
			}

			for (size_t i = 0; i < iteration; ++i)
			{
				static_cast<_Derived*>(this)->trainOne(pool, localData.data(), localRG.data());
				++iterated;
			}
		}

		double getLLPerWord() const override
		{
			return static_cast<const _Derived*>(this)->getLL() / words.size();
		}

		double getPerplexity() const override
		{
			return exp(-getLLPerWord());
		}

		std::vector<FLOAT> getWidsByTopic(TID tid) const
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

		std::vector<FLOAT> infer(DocumentBase* doc, size_t maxIter, FLOAT tolerance, FLOAT* ll = nullptr) const override
		{
			return static_cast<const _Derived*>(this)->infer(*static_cast<DocType*>(doc), maxIter, tolerance, ll);
		}

		FLOAT infer(const std::vector<DocumentBase*>& docs, size_t maxIter, FLOAT tolerance) const override
		{
			std::vector<DocType*> rDocs;
			std::transform(docs.begin(), docs.end(), std::back_inserter(rDocs), [](auto t) { return static_cast<DocType*>(t); });
			return static_cast<const _Derived*>(this)->infer(rDocs, maxIter, tolerance);
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