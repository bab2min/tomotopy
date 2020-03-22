#include <fstream>
#include <iostream>

#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif
#include "../TopicModel/TopicModel.hpp"
#include "docs.h"

void char2Byte(const std::string& str, std::vector<uint32_t>& startPos, std::vector<uint16_t>& length);

#define DEFINE_GETTER_PROTOTYPE(PREFIX, GETTER)\
PyObject* PREFIX##_##GETTER(TopicModelObject *self, void *closure);

#define DEFINE_GETTER(BASE, PREFIX, GETTER)\
PyObject* PREFIX##_##GETTER(TopicModelObject *self, void *closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		return py::buildPyValue(inst->GETTER());\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}\

#define DEFINE_SETTER_PROTOTYPE(PREFIX, SETTER)\
PyObject* PREFIX##_##SETTER(TopicModelObject *self, PyObject* val, void *closure);

#define DEFINE_SETTER_NON_NEGATIVE_INT(BASE, PREFIX, SETTER)\
int PREFIX##_##SETTER(TopicModelObject* self, PyObject* val, void* closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		auto v = PyLong_AsLong(val);\
		if (v == -1 && PyErr_Occurred()) throw bad_exception{};\
		if (v < 0) throw runtime_error{ #SETTER " must >= 0" };\
		inst->SETTER(v);\
	}\
	catch (const bad_exception&)\
	{\
		return -1;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return -1;\
	}\
	return 0;\
}


#define DEFINE_LOADER(PREFIX, TYPE) \
PyObject* PREFIX##_load(PyObject*, PyObject* args, PyObject *kwargs)\
{\
	const char* filename;\
	static const char* kwlist[] = { "filename", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename)) return nullptr;\
	try\
	{\
		ifstream str{ filename, ios_base::binary };\
		if (!str) throw runtime_error{ std::string("cannot open file '") + filename + std::string("'") };\
		for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)\
		{\
			str.seekg(0);\
			py::UniqueObj args = Py_BuildValue("(n)", i);\
			auto* p = PyObject_CallObject((PyObject*)&TYPE, args);\
			try\
			{\
				((TopicModelObject*)p)->inst->loadModel(str);\
			}\
			catch (const tomoto::serializer::UnfitException&)\
			{\
				Py_XDECREF(p);\
				continue;\
			}\
			((TopicModelObject*)p)->isPrepared = true;\
			return p;\
		}\
		throw runtime_error{ std::string("'") + filename + std::string("' is not valid model file") };\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}

#define DEFINE_DOCUMENT_GETTER_PROTOTYPE(NAME) \
PyObject* Document_##NAME(DocumentObject* self, void* closure);

#define DEFINE_DOCUMENT_GETTER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	try\
	{\
		if (!self->doc) throw runtime_error{ "doc is null!" };\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::one>*>(self->doc);\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::idf>*>(self->doc);\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::pmi>*>(self->doc);\
			if (doc) return py::buildPyValue(doc->FIELD);\
		} while (0);\
		throw runtime_error{ "doc doesn't has '" #FIELD "' field!" };\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}

#define DEFINE_DOCUMENT_GETTER_REORDER(DOCTYPE, NAME, FIELD) \
PyObject* Document_##NAME(DocumentObject* self, void* closure)\
{\
	try\
	{\
		if (!self->doc) throw runtime_error{ "doc is null!" };\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::one>*>(self->doc);\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::idf>*>(self->doc);\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		do\
		{\
			auto* doc = dynamic_cast<const DOCTYPE<tomoto::TermWeight::pmi>*>(self->doc);\
			if (doc) return buildPyValueReorder(doc->FIELD, doc->wOrder);\
		} while (0);\
		throw runtime_error{ "doc doesn't has '" #FIELD "' field!" };\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
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


extern PyObject* gModule;
extern PyTypeObject Document_type, Corpus_type, Dictionary_type;
extern PyTypeObject LDA_type;
extern PyTypeObject DMR_type;
extern PyTypeObject HDP_type;
extern PyTypeObject MGLDA_type;
extern PyTypeObject PA_type;
extern PyTypeObject HPA_type;
extern PyTypeObject CT_type;
extern PyTypeObject SLDA_type;
extern PyTypeObject HLDA_type;
extern PyTypeObject LLDA_type;
extern PyTypeObject PLDA_type;

struct TopicModelObject
{
	PyObject_HEAD;
	tomoto::ITopicModel* inst;
	bool isPrepared;
	size_t minWordCnt, minWordDf;
	size_t removeTopWord;

	static void dealloc(TopicModelObject* self);
};


struct CorpusObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;

	static void dealloc(CorpusObject* self);
};


struct DocumentObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;
	const tomoto::DocumentBase* doc;
	bool owner;

	static PyObject* repr(DocumentObject* self);

	static void dealloc(DocumentObject* self);
};

struct DictionaryObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;
	const tomoto::Dictionary* dict;

	static void dealloc(DictionaryObject* self);

	static Py_ssize_t len(DictionaryObject* self);

	static PyObject* getitem(DictionaryObject* self, Py_ssize_t key);

	static PyObject* repr(DictionaryObject* self);

	static int init(DictionaryObject *self, PyObject *args, PyObject *kwargs);
};

DEFINE_GETTER_PROTOTYPE(LDA, getK);
DEFINE_GETTER_PROTOTYPE(LDA, getAlpha);
DEFINE_GETTER_PROTOTYPE(LDA, getEta);

PyObject* Document_LDA_Z(DocumentObject* self, void* closure);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(metadata);

PyObject* Document_HDP_Z(DocumentObject* self, void* closure);

PyObject* Document_HLDA_Z(DocumentObject* self, void* closure);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(windows);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(Z2);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(beta);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(y);

DEFINE_DOCUMENT_GETTER_PROTOTYPE(labels);

PyObject* Document_getSubTopics(DocumentObject* self, PyObject* args, PyObject *kwargs);
PyObject* Document_getSubTopicDist(DocumentObject* self);

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

static const char* corpus_feeder_name = "_feed_docs_to";