#pragma once

#include "../TopicModel/LDA.h"
#include "../Utils/Dictionary.h"
#include "module.h"
#include "../Labeling/Phraser.hpp"

extern PyTypeObject UtilsCorpus_type;
extern PyTypeObject UtilsCorpusIter_type;
extern PyTypeObject UtilsDocument_type;
extern PyTypeObject UtilsVocab_type;
extern PyTypeObject Phraser_type;

struct VocabObject
{
	PyObject_HEAD;
	tomoto::Dictionary* vocabs;
	PyObject* dep;
	size_t size;

	static int init(VocabObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(VocabObject* self);
	static PyObject* getstate(VocabObject* self, PyObject*);
	static PyObject* setstate(VocabObject* self, PyObject* args);
	
	static Py_ssize_t len(VocabObject* self);
	static PyObject* getitem(VocabObject* self, Py_ssize_t key);
	static PyObject* repr(VocabObject* self);
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

struct CorpusObject
{
	PyObject_HEAD;
	union
	{
		std::vector<tomoto::RawDoc> docs;
		std::vector<size_t> docIdcs;
		std::vector<std::shared_ptr<tomoto::DocumentBase>> docsMade;
	};
	std::unordered_map<std::string, size_t> invmap;
	union
	{
		PyObject* depObj;
		VocabObject* vocab;
		TopicModelObject* tm;
	};
	bool made;

	inline bool isIndependent() const
	{
		return vocab && !!PyObject_TypeCheck(vocab, &UtilsVocab_type);
	}

	inline bool isSubDocs() const
	{
		return !docIdcs.empty() && !invmap.empty();
	}

	size_t findUid(const std::string& uid) const;
	const tomoto::RawDocKernel* getDoc(size_t idx) const;

	static CorpusObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs);
	static int init(CorpusObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(CorpusObject* self);
	static PyObject* getstate(CorpusObject* self, PyObject*);
	static PyObject* setstate(CorpusObject* self, PyObject* args);
	static PyObject* addDoc(CorpusObject* self, PyObject* args, PyObject* kwargs);
	static PyObject* extractNgrams(CorpusObject* self, PyObject* args, PyObject* kwargs);
	static PyObject* concatNgrams(CorpusObject* self, PyObject* args, PyObject* kwargs);
	static Py_ssize_t len(CorpusObject* self);
	static PyObject* getitem(CorpusObject* self, PyObject* idx);

	static PyObject* iter(CorpusObject* self);

	const tomoto::Dictionary& getVocabDict() const;
};

struct CorpusIterObject
{
	PyObject_HEAD;
	CorpusObject* corpus;
	size_t idx;
	static int init(CorpusIterObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(CorpusIterObject* self);

	static CorpusIterObject* iter(CorpusIterObject* self);
	static PyObject* iternext(CorpusIterObject* self);
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

struct DocumentObject
{
	PyObject_HEAD;
	const tomoto::RawDocKernel* doc;
	CorpusObject* corpus;
	bool owner;
	bool initialized;

	inline const tomoto::RawDoc* getRawDoc() const { return (const tomoto::RawDoc*)doc; }
	inline const tomoto::DocumentBase* getBoundDoc() const { return (const tomoto::DocumentBase*)doc; }

	static int init(DocumentObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(DocumentObject* self);

	static Py_ssize_t len(DocumentObject* self);
	static PyObject* repr(DocumentObject* self);
	static PyObject* getitem(DocumentObject* self, Py_ssize_t idx); // access to words
	static PyObject* getWords(DocumentObject* self, void* closure); // access to word ids
	static PyObject* getRaw(DocumentObject* self, void* closure);
	static PyObject* getSpan(DocumentObject* self, void* closure);
	static PyObject* getWeight(DocumentObject* self, void* closure);
	static PyObject* getUid(DocumentObject* self, void* closure);

	static PyObject* getattro(DocumentObject* self, PyObject* attr);
};

struct PhraserObject
{
	PyObject_HEAD;

	tomoto::Dictionary vocabs;
	using TrieNode = tomoto::Trie<tomoto::Vid, size_t, tomoto::ConstAccess<tomoto::phraser::map<tomoto::Vid, int32_t>>>;
	std::vector<TrieNode> trie_nodes;
	std::vector<std::pair<std::string, size_t>> cand_info;

	static PhraserObject* _new(PyTypeObject* subtype, PyObject* args, PyObject* kwargs);
	static int init(PhraserObject* self, PyObject* args, PyObject* kwargs);
	static void dealloc(PhraserObject* self);
	static PyObject* repr(PhraserObject* self);
	static PyObject* call(PhraserObject* self, PyObject* args, PyObject* kwargs);
	static PyObject* save(PhraserObject* self, PyObject* args, PyObject* kwargs);
	static PyObject* load(PhraserObject*, PyObject* args, PyObject* kwargs);
	static PyObject* findall(PhraserObject* self, PyObject* args, PyObject* kwargs);
};

void addUtilsTypes(PyObject* gModule);

template<
	template<tomoto::TermWeight tw> class DocTy, 
	typename Fn
>
PyObject* docVisit(tomoto::DocumentBase* doc, Fn&& visitor)
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

	return nullptr;
}

template<
	template<tomoto::TermWeight tw> class DocTy,
	typename Fn
>
PyObject* docVisit(const tomoto::DocumentBase* doc, Fn&& visitor)
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

	return nullptr;
}

#define DEFINE_DOCUMENT_GETTER_PROTOTYPE(NAME) \
PyObject* Document_##NAME(DocumentObject* self, void* closure);

#define DEFINE_DOCUMENT_GETTER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	return py::handleExc([&]()\
	{\
		if (self->corpus->isIndependent()) throw py::AttributeError{ "doc has no `" #FIELD "` field!" };\
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };\
		if (auto* ret = docVisit<DOCTYPE>(self->getBoundDoc(), [](auto* doc)\
		{\
			return py::buildPyValue(doc->FIELD);\
		})) return ret;\
		throw py::AttributeError{ "doc has no `" #FIELD "` field!" };\
	});\
}

#define DEFINE_DOCUMENT_GETTER_WITHOUT_EXC(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	return py::handleExc([&]()\
	{\
		if (self->corpus->isIndependent()) throw py::AttributeError{ "doc has no `" #FIELD "` field!" };\
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };\
		return docVisit<DOCTYPE>(self->getBoundDoc(), [](auto* doc)\
		{\
			return py::buildPyValue(doc->FIELD);\
		});\
	});\
}

#define DEFINE_DOCUMENT_GETTER_REORDER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	return py::handleExc([&]()\
	{\
		if (self->corpus->isIndependent()) throw py::AttributeError{ "doc has no `" #FIELD "` field!" };\
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };\
		if (auto* ret = docVisit<DOCTYPE>(self->getBoundDoc(), [](auto* doc)\
		{\
			return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		})) return ret;\
		throw py::AttributeError{ "doc has no `" #FIELD "` field!" }; \
	});\
}

namespace py
{
	template<typename _Ty>
	PyObject* buildPyValue(const tomoto::tvector<_Ty>& v)
	{
		auto ret = PyList_New(v.size());
		size_t id = 0;
		for (auto& e : v)
		{
			PyList_SetItem(ret, id++, buildPyValue(e));
		}
		return ret;
	}
}

PyObject* Document_LDA_Z(DocumentObject* self, void* closure);

PyObject* Document_HDP_Z(DocumentObject* self, void* closure);

PyObject* Document_HLDA_Z(DocumentObject* self, void* closure);

PyObject* Document_DMR_metadata(DocumentObject* self, void* closure);
PyObject* Document_DMR_multiMetadata(DocumentObject* self, void* closure);

PyObject* Document_numericMetadata(DocumentObject* self, void* closure);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(windows);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(Z2);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(path);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(beta);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(y);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(labels);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(eta);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(timepoint);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(pseudo_doc_id);

PyObject* Document_getSubTopics(DocumentObject* self, PyObject* args, PyObject* kwargs);
PyObject* Document_getSubTopicDist(DocumentObject* self, PyObject* args, PyObject* kwargs);

PyObject* Document_getCountVector(DocumentObject* self);

PyObject* Document_getTopicsFromPseudoDoc(DocumentObject* self, size_t topN);
PyObject* Document_getTopicDistFromPseudoDoc(DocumentObject* self, bool normalize);


template<typename _Target, typename _Order>
PyObject* buildPyValueReorder(const _Target& target, const _Order& order)
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
PyObject* buildPyValueReorder(const _Target& target, const _Order& order, _Tx&& transformer)
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

std::vector<size_t> insertCorpus(TopicModelObject* self, PyObject* corpus, PyObject* transform);
CorpusObject* makeCorpus(TopicModelObject* self, PyObject* _corpus, PyObject* transform);
