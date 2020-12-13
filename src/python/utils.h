#pragma once

#include "../Utils/Dictionary.h"
#include "module.h"

extern PyTypeObject UtilsCorpus_type;
extern PyTypeObject UtilsCorpusIter_type;
extern PyTypeObject UtilsDocument_type;
extern PyTypeObject UtilsVocab_type;

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
- Bound Model: used for tomotopy.*Model.docs
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
		return !!PyObject_TypeCheck(vocab, &UtilsVocab_type);
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

	ssize_t operator-(const DocWordIterator& o) const
	{
		return (ssize_t)idx - (ssize_t)o.idx;
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

void addUtilsTypes(PyObject* gModule);

#define DEFINE_DOCUMENT_GETTER_PROTOTYPE(NAME) \
PyObject* Document_##NAME(DocumentObject* self, void* closure);

#define DEFINE_DOCUMENT_GETTER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	try\
	{\
		if (self->corpus->isIndependent()) throw runtime_error{ "doc doesn't has `" #FIELD "` field!" };\
		if (!self->doc) throw runtime_error{ "doc is null!" };\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::one>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::idf>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::pmi>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		throw runtime_error{ "doc doesn't has `" #FIELD "` field!" };\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_AttributeError, e.what());\
		return nullptr;\
	}\
}

#define DEFINE_DOCUMENT_GETTER_WITHOUT_EXC(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	try\
	{\
		if (self->corpus->isIndependent()) throw runtime_error{ "doc doesn't has `" #FIELD "` field!" };\
		if (!self->doc) throw runtime_error{ "doc is null!" };\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::one>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::idf>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::pmi>*>(self->getBoundDoc());\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		return nullptr;\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_AttributeError, e.what());\
		return nullptr;\
	}\
}

#define DEFINE_DOCUMENT_GETTER_REORDER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	try\
	{\
	if (self->corpus->isIndependent()) throw runtime_error{ "doc doesn't has `" #FIELD "` field!" };\
		if (!self->doc) throw runtime_error{ "doc is null!" };\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::one>*>(self->getBoundDoc());\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::idf>*>(self->getBoundDoc());\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::pmi>*>(self->getBoundDoc());\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		throw runtime_error{ "doc doesn't has `" #FIELD "` field!" };\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_AttributeError, e.what());\
		return nullptr;\
	}\
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

PyObject* Document_GDMR_metadata(DocumentObject* self, void* closure);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(windows);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(Z2);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(path);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(beta);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(y);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(labels);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(eta);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(timepoint);

PyObject* Document_getSubTopics(DocumentObject* self, PyObject* args, PyObject* kwargs);
PyObject* Document_getSubTopicDist(DocumentObject* self);

PyObject* Document_getCountVector(DocumentObject* self);

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