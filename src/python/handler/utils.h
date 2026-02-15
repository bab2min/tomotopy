#pragma once

#include "../../TopicModel/LDA.h"
#include "../../Utils/Dictionary.h"
#include "module.h"
#include "../../Labeling/Phraser.hpp"

struct VocabObject : public py::CObject<VocabObject>
{
	tomoto::Dictionary* vocabs = nullptr;
	py::UniqueObj dep;
	size_t size = 0;

	VocabObject() = default;
	~VocabObject();

	py::UniqueObj getstate() const;
	void setstate(PyObject* args);
	
	size_t len() const;
	std::string_view getitem(size_t key) const;
	py::UniqueObj repr() const;
};

/*
Two modes of CorpusObject
- Independent Mode: used for tomotopy.utils.Corpus
	- std::vector<tomoto::RawDoc> docs;
	- std::unordered_map<std::string, size_t> invmap;
    - VocabObject* vocab;
- Bound Model: used for tomotopy.***Model.docs
    - std::vector<size_t> docIdcs;
	- std::unordered_map<std::string, size_t> invmap;
    - TopicModelObject* tm;
*/

struct DocumentObject;
struct CorpusIterObject;

struct CorpusObject : public py::CObject<CorpusObject>
{
	union
	{
		std::vector<tomoto::RawDoc> docs;
		std::vector<size_t> docIdcs;
		std::vector<std::shared_ptr<tomoto::DocumentBase>> docsMade;
	};
	std::unordered_map<std::string, size_t> invmap;
	union
	{
		py::UniqueObj depObj;
		py::UniqueCObj<VocabObject> vocab;
		py::UniqueCObj<LDAModelObject> tm;
	};
	bool made = false;

	inline bool isIndependent() const
	{
		return vocab && !!PyObject_TypeCheck(vocab.get(), py::Type<VocabObject>);
	}

	inline bool isSubDocs() const
	{
		return !docIdcs.empty() && !invmap.empty();
	}

	size_t findUid(const std::string& uid) const;
	const tomoto::RawDocKernel* getDoc(size_t idx) const;

	CorpusObject();
	CorpusObject(PyObject* vocabInst, PyObject* dep);
	~CorpusObject();

	py::UniqueObj getstate() const;
	void setstate(PyObject* args);
	size_t addDoc(PyObject* words, PyObject* raw, PyObject* userData, PyObject* additionalKwargs);
	size_t addDocs(PyObject* tokenizedIter, PyObject* rawIter, PyObject* metadataIter);
	py::UniqueObj extractNgrams(size_t minCf = 10, size_t minDf = 5, size_t maxLen = 5, size_t maxCand = 5000,
		float minScore = -INFINITY, bool normalized = false, size_t workers = 1) const;
	size_t concatNgrams(PyObject* cands, const std::string& delimiter = "_");
	size_t len() const;
	py::UniqueObj getitem(PyObject* idx) const;

	py::UniqueCObj<CorpusIterObject> iter();

	const tomoto::Dictionary& getVocabDict() const;
};

struct CorpusIterObject : public py::CObject<CorpusIterObject>
{
	py::UniqueCObj<CorpusObject> corpus;
	size_t idx = 0;

	CorpusIterObject() = default;
	~CorpusIterObject() = default;

	py::UniqueCObj<CorpusIterObject> iter();
	py::UniqueCObj<DocumentObject> iternext();
};

class DocWordIterator
{
	const tomoto::Vid* base = nullptr;
	const uint32_t* order = nullptr;
	size_t idx = 0;
public:
	DocWordIterator(const tomoto::Vid* _base = nullptr,
		const uint32_t* _order = nullptr,
		size_t _idx = 0)
		: base{ _base }, order{ _order }, idx{ _idx }
	{
	}

	DocWordIterator(const DocWordIterator&) = default;
	DocWordIterator(DocWordIterator&&) = default;

	DocWordIterator& operator++()
	{
		++idx;
		return *this;
	}

	const tomoto::Vid& operator*() const
	{
		if (order) return base[order[idx]];
		return base[idx];
	}

	const tomoto::Vid& operator[](int i) const
	{
		if (order) return base[order[idx + i]];
		return base[idx + i];
	}

	bool operator==(const DocWordIterator& o) const
	{
		return idx == o.idx && base == o.base && order == o.order;
	}

	bool operator!=(const DocWordIterator& o) const
	{
		return !operator==(o);
	}

	std::ptrdiff_t operator-(const DocWordIterator& o) const
	{
		return (std::ptrdiff_t)idx - (std::ptrdiff_t)o.idx;
	}
};

DocWordIterator wordBegin(const tomoto::RawDocKernel* doc, bool independent);
DocWordIterator wordEnd(const tomoto::RawDocKernel* doc, bool independent);

struct DocumentObject : py::CObject<DocumentObject>
{
	const tomoto::RawDocKernel* doc = nullptr;
	py::UniqueCObj<CorpusObject> corpus;
	bool owner = false;
	bool initialized = false;

	inline const tomoto::RawDoc* getRawDoc() const { return (const tomoto::RawDoc*)doc; }
	inline const tomoto::DocumentBase* getBoundDoc() const { return (const tomoto::DocumentBase*)doc; }

	DocumentObject() = default;
	DocumentObject(PyObject* corpus);
	~DocumentObject();

	size_t len() const;
	std::string repr() const;
	std::optional<std::string_view> getitem(size_t idx) const; // access to words
	py::UniqueObj getAllWords() const; // access to word ids
	const tomoto::SharedString& getRaw() const;
	py::UniqueObj getSpan() const;
	tomoto::Float getWeight() const;
	const tomoto::SharedString& getUid() const;

	py::UniqueObj getattro(PyObject* attr) const;

	std::vector<std::pair<tomoto::Tid, tomoto::Float>> getTopics(size_t topN, bool fromPseudoDoc = false) const;
	std::vector<float> getTopicDist(bool normalize = true, bool fromPseudoDoc = false) const;
	std::vector<std::pair<std::string, tomoto::Float>> getWords(size_t topN) const;
	py::UniqueObj getZ() const;
	std::string_view getMetadata() const;
	double getLL() const;

	// for LDAModel
	py::UniqueObj getZFromLDA() const;
	std::vector<float> getCountVector() const;

	// for DMRModel
	std::optional<std::string_view> getMetadataFromDMR() const;
	py::UniqueObj getMultiMetadata() const;

	// for GDMRModel
	std::optional<std::vector<float>> getNumericMetadata() const;

	// for PAModel
	py::UniqueObj getZ2() const;
	std::vector<std::pair<tomoto::Tid, tomoto::Float>> getSubTopics(size_t topN = 10) const;
	std::vector<float> getSubTopicDist(bool normalize = true) const;

	// for MGLDAModel
	py::UniqueObj getWindows() const;

	// for CTModel
	py::UniqueObj getBeta() const;

	// for HDPModel
	py::UniqueObj getZFromHDP() const;

	// for HLDAModel
	py::UniqueObj getZFromHLDA() const;
	std::vector<int32_t> getPath() const;

	// for SLDAModel
	py::UniqueObj getY() const;

	// for LLDAModel
	py::UniqueObj getLabels() const;

	// for DTModel
	py::UniqueObj getEta() const;
	size_t getTimepoint() const;

	// for PTModel
	size_t getPseudoDocId() const;
	std::vector<std::pair<tomoto::Tid, tomoto::Float>> getTopicsFromPseudoDoc(size_t topN) const;
	std::vector<float> getTopicDistFromPseudoDoc(bool normalize = true) const;
};

struct PhraserObject : public py::CObject<PhraserObject>
{
	using TrieNode = tomoto::Trie<tomoto::Vid, size_t, tomoto::ConstAccess<tomoto::phraser::map<tomoto::Vid, int32_t>>>;

	tomoto::Dictionary vocabs;
	std::vector<TrieNode> trie_nodes;
	std::vector<std::pair<std::string, size_t>> cand_info;

	PhraserObject();
	PhraserObject(PyObject* candidates, const std::string& delimiter);
	~PhraserObject() = default;

	std::string repr() const;
	py::UniqueObj call(PyObject* words) const;
	py::UniqueObj findall(PyObject* words) const;
	void save(const std::string& path) const;
	static py::UniqueCObj<PhraserObject> load(PyObject* cls, const std::string& path);
};

void addUtilsTypes(py::Module& module);

template<
	template<tomoto::TermWeight tw> class DocTy, 
	typename Fn
>
auto docVisit(tomoto::DocumentBase* doc, Fn&& visitor) -> decltype(visitor(static_cast<DocTy<tomoto::TermWeight::one>*>(doc)))
{
	if (auto* d = dynamic_cast<DocTy<tomoto::TermWeight::one>*>(doc))
	{
		return visitor(d);
	}

	if (auto* d = dynamic_cast<DocTy<tomoto::TermWeight::idf>*>(doc))
	{
		return visitor(d);
	}

	if (auto* d = dynamic_cast<DocTy<tomoto::TermWeight::pmi>*>(doc))
	{
		return visitor(d);
	}
	return {};
}

template<
	template<tomoto::TermWeight tw> class DocTy,
	typename Fn
>
auto docVisit(const tomoto::DocumentBase* doc, Fn&& visitor) -> decltype(visitor(static_cast<const DocTy<tomoto::TermWeight::one>*>(doc)))
{
	if (auto* d = dynamic_cast<const DocTy<tomoto::TermWeight::one>*>(doc))
	{
		return visitor(d);
	}

	if (auto* d = dynamic_cast<const DocTy<tomoto::TermWeight::idf>*>(doc))
	{
		return visitor(d);
	}

	if (auto* d = dynamic_cast<const DocTy<tomoto::TermWeight::pmi>*>(doc))
	{
		return visitor(d);
	}

	return {};
}

namespace py
{
	template<typename _Ty>
	py::UniqueObj buildPyValue(const tomoto::tvector<_Ty>& v)
	{
		auto ret = py::UniqueObj{ PyList_New(v.size()) };
		size_t id = 0;
		for (auto& e : v)
		{
			PyList_SetItem(ret.get(), id++, buildPyValue(e).release());
		}
		return ret;
	}
}

template<typename _Target, typename _Order>
py::UniqueObj buildPyValueReorder(const _Target& target, const _Order& order)
{
	if (order.empty())
	{
		return py::buildPyValue(target);
	}
	else
	{
		using _OType = decltype(order[0]);
		return py::buildPyValueTransform(order.begin(), order.end(), [&](_OType idx)
		{
			return target[idx];
		});
	}
}

template<typename _Target, typename _Order, typename _Tx>
py::UniqueObj buildPyValueReorder(const _Target& target, const _Order& order, _Tx&& transformer)
{
	if (order.empty())
	{
		return py::buildPyValueTransform(target.begin(), target.end(), transformer);
	}
	else
	{
		using _OType = decltype(order[0]);
		return py::buildPyValueTransform(order.begin(), order.end(), [&](_OType idx)
		{
			return transformer(target[idx]);
		});
	}
}
