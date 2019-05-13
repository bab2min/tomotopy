#include <fstream>
#include <iostream>

#include "Utils/Utils.hpp"
#include "TopicModel/TopicModel.hpp"
#include "TopicModel/LDAModel.hpp"
#include "TopicModel/MGLDAModel.hpp"
#include "TopicModel/HDPModel.hpp"
#include "TopicModel/DMRModel.hpp"
#include "TopicModel/PAModel.hpp"
#include "TopicModel/HPAModel.hpp"
#include "TopicModel/CIDMRModel.hpp"

#ifdef _DEBUG
#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0);
#include "PyUtils.h"
#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

#include "pythonDocs.h"

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
#define DEFINE_GETTER(BASE, PREFIX, GETTER)\
static PyObject* PREFIX##_##GETTER(TopicModelObject *self, void *closure)\
{\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<BASE*>(self->inst);\
		return py::buildPyValue(inst->GETTER());\
	}\
	catch (const bad_exception& e)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}\

#define DEFINE_LOADER(PREFIX, TYPE) \
static PyObject* PREFIX##_load(PyObject*, PyObject* args, PyObject *kwargs)\
{\
	const char* filename;\
	static const char* kwlist[] = { "filename", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &filename)) return nullptr;\
	try\
	{\
		ifstream str{ filename, ios_base::binary };\
		if (!str) throw runtime_error{ "cannot open file '"s + filename + "'"s };\
		for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)\
		{\
			str.seekg(0);\
			auto* p = PyObject_CallObject((PyObject*)&TYPE, Py_BuildValue("(n)", i));\
			try\
			{\
				((TopicModelObject*)p)->inst->loadModel(str);\
			}\
			catch (const tomoto::serializer::UnfitException& e)\
			{\
				Py_XDECREF(p);\
				continue;\
			}\
			((TopicModelObject*)p)->isPrepared = true;\
			return p;\
		}\
		throw runtime_error{ "'"s + filename + "' is not valid model file"s };\
	}\
	catch (const bad_exception& e)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}

using namespace std;

static PyObject* gModule;

struct TopicModelObject
{
	PyObject_HEAD;
	tomoto::ITopicModel* inst;
	bool isPrepared;

	static void dealloc(TopicModelObject* self)
	{
		DEBUG_LOG("TopicModelObject Dealloc " << self);
		if (self->inst)
		{
			delete self->inst;
		}
		Py_TYPE(self)->tp_free((PyObject*)self);
	}
};


struct CorpusObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;

	static void dealloc(CorpusObject* self)
	{
		DEBUG_LOG("CorpusObject Dealloc " << self->parentModel->ob_base.ob_type << ", " << self->parentModel->ob_base.ob_refcnt);
		Py_XDECREF(self->parentModel);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}
};


struct DocumentObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;
	const tomoto::DocumentBase* doc;
	bool owner;

	static void dealloc(DocumentObject* self)
	{
		DEBUG_LOG("DocumentObject Dealloc " << self->parentModel->ob_base.ob_type << ", " << self->parentModel->ob_base.ob_refcnt);
		Py_XDECREF(self->parentModel);
		if (self->owner)
		{
			delete self->doc;
		}
		Py_TYPE(self)->tp_free((PyObject*)self);
	}
};

struct DictionaryObject
{
	PyObject_HEAD;
	TopicModelObject* parentModel;
	const tomoto::Dictionary* dict;

	static void dealloc(DictionaryObject* self)
	{
		Py_XDECREF(self->parentModel);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static Py_ssize_t len(DictionaryObject* self)
	{
		try
		{
			if (!self->dict) throw runtime_error{ "dict is null" };
			return self->dict->size();
		}
		catch (const bad_exception& e)
		{
			return -1;
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
	}

	static PyObject* getitem(DictionaryObject* self, Py_ssize_t key)
	{
		try
		{
			if (!self->dict) throw runtime_error{ "inst is null" };
			if (key >= self->dict->size())
			{
				PyErr_SetString(PyExc_IndexError, "");
				throw bad_exception{};
			}
			return py::buildPyValue(self->dict->toWord(key));
		}
		catch (const bad_exception& e)
		{
			return nullptr;
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return nullptr;
		}
	}

	static int init(DictionaryObject *self, PyObject *args, PyObject *kwargs)
	{
		PyObject* argParent;
		const tomoto::Dictionary* dict;
		static const char* kwlist[] = { "f", "g", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "On", (char**)kwlist, &argParent, &dict)) return -1;
		try
		{
			self->parentModel = (TopicModelObject*)argParent;
			Py_INCREF(argParent);
			self->dict = dict;
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
		return 0;
	}
};

static PySequenceMethods Dictionary_seq_methods = {
	(lenfunc)DictionaryObject::len,
	nullptr,
	nullptr,
	(ssizeargfunc)DictionaryObject::getitem,
};

static PyTypeObject Dictionary_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.Dictionary",             /* tp_name */
	sizeof(DictionaryObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)DictionaryObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	&Dictionary_seq_methods,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,   /* tp_flags */
	"`list`-like Dictionary interface for vocabularies",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	PySeqIter_New, /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)DictionaryObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


static PyObject* Document_getTopics(DocumentObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topN = 10;
	static const char* kwlist[] = { "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|n", (char**)kwlist, &topN)) return nullptr;
	try
	{
		if (!self->parentModel->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->parentModel->inst->getTopicsByDocSorted(self->doc, topN));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_getTopicDist(DocumentObject* self)
{
	try
	{
		if (!self->parentModel->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->parentModel->inst->getTopicsByDoc(self->doc));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}


static PyMethodDef Document_methods[] =
{
	{ "get_topics", (PyCFunction)Document_getTopics, METH_VARARGS | METH_KEYWORDS, Document_get_topics__doc__ },
	{ "get_topic_dist", (PyCFunction)Document_getTopicDist, METH_NOARGS, Document_get_topic_dist__doc__ },
	{ nullptr }
};


static int Document_init(DocumentObject *self, PyObject *args, PyObject *kwargs)
{
	PyObject* argParent;
	size_t docId, owner = 0;
	static const char* kwlist[] = { "f", "g", "h", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Onn", (char**)kwlist, &argParent, &docId, &owner)) return -1;
	try
	{
		self->parentModel = (TopicModelObject*)argParent;
		if (owner)
		{
			self->doc = (tomoto::DocumentBase*)docId;
			self->owner = true;
		}
		else
		{
			self->doc = self->parentModel->inst->getDoc(docId);
			self->owner = false;
		}
		Py_INCREF(argParent);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* Document_words(DocumentObject* self, void* closure)
{
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		return py::buildPyValue(self->doc->words);
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_weight(DocumentObject* self, void* closure)
{
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		return py::buildPyValue(self->doc->weight);
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

#define DEFINE_DOCUMENT_GETTER(DOCTYPE, NAME, FIELD) \
static PyObject* Document_##NAME(DocumentObject* self, void* closure)\
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
	catch (const bad_exception& e)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}

static PyObject* Document_Z(DocumentObject* self, void* closure)
{
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentHDP<tomoto::TermWeight::one>*>(self->doc);
			if (doc) return py::buildPyValueTransform(doc->Zs.begin(), doc->Zs.end(), [doc](auto x) { return doc->numTopicByTable[x].topic; });
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentHDP<tomoto::TermWeight::idf>*>(self->doc);
			if (doc) return py::buildPyValueTransform(doc->Zs.begin(), doc->Zs.end(), [doc](auto x) { return doc->numTopicByTable[x].topic; });
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentHDP<tomoto::TermWeight::pmi>*>(self->doc);
			if (doc) return py::buildPyValueTransform(doc->Zs.begin(), doc->Zs.end(), [doc](auto x) { return doc->numTopicByTable[x].topic; });
		} while (0);

		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::one>*>(self->doc);
			if (doc) return py::buildPyValue(doc->Zs);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::idf>*>(self->doc);
			if (doc) return py::buildPyValue(doc->Zs);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::pmi>*>(self->doc);
			if (doc) return py::buildPyValue(doc->Zs);
		} while (0);
		throw runtime_error{ "doc doesn't has 'Zs' field!" };
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

DEFINE_DOCUMENT_GETTER(tomoto::DocumentDMR, metadata, metadata);
DEFINE_DOCUMENT_GETTER(tomoto::DocumentMGLDA, windows, Vs);
DEFINE_DOCUMENT_GETTER(tomoto::DocumentPA, Z2, Z2s);

static PyGetSetDef Document_getseters[] = {
	{ (char*)"words", (getter)Document_words, nullptr, (char*)"word ids", NULL },
	{ (char*)"weight", (getter)Document_weight, nullptr, (char*)"weights for each word", NULL },
	{ (char*)"topics", (getter)Document_Z, nullptr, (char*)"topics for each word", NULL },
	{ (char*)"metadata", (getter)Document_metadata, nullptr, (char*)"metadata of document (for only `tomotopy.DMR` model)", NULL },
	{ (char*)"subtopics", (getter)Document_Z2, nullptr, (char*)"sub topics for each word (for only `tomotopy.PA` and `tomotopy.HPA` model)", NULL },
	{ (char*)"windows", (getter)Document_windows, nullptr, (char*)"window ids for each word (for only `tomotopy.MGLDA` model)", NULL },
	{ nullptr },
};


static PyTypeObject Document_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.Document",             /* tp_name */
	sizeof(DocumentObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)DocumentObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,   /* tp_flags */
	Document___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	Document_methods,             /* tp_methods */
	0,						 /* tp_members */
	Document_getseters,        /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)Document_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static Py_ssize_t Corpus_len(CorpusObject* self)
{
	try
	{
		if (!self->parentModel->inst) throw runtime_error{ "inst is null" };
		return self->parentModel->inst->getNumDocs();
	}
	catch (const bad_exception& e)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

static PyObject* Corpus_getitem(CorpusObject* self, Py_ssize_t key)
{
	try
	{
		if (!self->parentModel->inst) throw runtime_error{ "inst is null" };
		if (key >= self->parentModel->inst->getNumDocs())
		{
			PyErr_SetString(PyExc_IndexError, "");
			throw bad_exception{};
		}
		return PyObject_CallObject((PyObject*)&Document_type, Py_BuildValue("(Nnn)", self->parentModel, key, 0));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static int Corpus_init(CorpusObject *self, PyObject *args, PyObject *kwargs)
{
	PyObject* argParent;
	static const char* kwlist[] = { "f", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argParent)) return -1;
	try
	{
		self->parentModel = (TopicModelObject*)argParent;
		Py_INCREF(argParent);
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PySequenceMethods Corpus_seq_methods = {
	(lenfunc)Corpus_len,
	nullptr, 
	nullptr,
	(ssizeargfunc)Corpus_getitem,
};

static PyTypeObject Corpus_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.Corpus",             /* tp_name */
	sizeof(CorpusObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CorpusObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	&Corpus_seq_methods,       /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT,   /* tp_flags */
	"Corpus type",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	PySeqIter_New,              /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)Corpus_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static PyObject* LDA_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* DMR_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* HDP_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* MGLDA_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* PA_load(PyObject*, PyObject* args, PyObject *kwargs);
static PyObject* HPA_load(PyObject*, PyObject* args, PyObject *kwargs);

static int LDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "k", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnffn", (char**)kwlist, &tw, &K, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, K, alpha, eta, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* LDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->addDoc(py::makeIterToVector(iter));
		return py::buildPyValue(ret);
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->makeDoc(py::makeIterToVector(iter));
		return PyObject_CallObject((PyObject*)&Document_type, Py_BuildValue("(Nnn)", self, ret.release(), 1));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_train(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t iteration = 1, workers = 0;
	static const char* kwlist[] = { "iter", "workers", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nn", (char**)kwlist, &iteration, &workers)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		inst->train(iteration, workers);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{"must topic_id < K"};
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < K" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWidsByTopic(topicId));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argDoc, *iter = nullptr, *item;
	size_t iteration = 100;
	float tolerance = -1;
	static const char* kwlist[] = { "document", "iter", "tolerance", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nf", (char**)kwlist, &argDoc, &iteration, &tolerance)) return nullptr;
	DEBUG_LOG("infer " << self->ob_base.ob_type << ", " << self->ob_base.ob_refcnt);
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if ((iter = PyObject_GetIter(argDoc)) != nullptr)
		{
			py::AutoReleaser arIter{ iter };
			std::vector<tomoto::DocumentBase*> docs;
			while (item = PyIter_Next(iter))
			{
				py::AutoReleaser arItem{ item };
				if (Py_TYPE(item) != &Document_type) throw runtime_error{ "document must be tomotopy.Document type or list of tomotopy.Document" };
				auto* doc = (DocumentObject*)item;
				if (doc->parentModel != self) throw runtime_error{ "document was from another model, not fit to this model" };
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
			}
			if (PyErr_Occurred()) throw bad_exception{};
			if (!self->isPrepared)
			{
				self->inst->prepare();
				self->isPrepared = true;
			}
			auto ll = self->inst->infer(docs, iteration, tolerance);
			PyObject* ret = PyList_New(docs.size());
			size_t i = 0;
			for (auto d : docs)
			{
				PyList_SetItem(ret, i++, py::buildPyValue(self->inst->getTopicsByDoc(d)));
			}
			return Py_BuildValue("(Nf)", ret, ll);
		}
		else
		{
			PyErr_Clear();
			if (Py_TYPE(argDoc) != &Document_type) throw runtime_error{ "document must be tomotopy.Document type or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)argDoc;
			if (doc->parentModel != self) throw runtime_error{ "document was from another model, not fit to this model" };
			if (!self->isPrepared)
			{
				self->inst->prepare();
				self->isPrepared = true;
			}

			if (doc->owner)
			{
				float ll = self->inst->infer((tomoto::DocumentBase*)doc->doc, iteration, tolerance);
				return Py_BuildValue("(Nf)", py::buildPyValue(self->inst->getTopicsByDoc(doc->doc)), ll);
			}
			else
			{
				return Py_BuildValue("(Ns)", py::buildPyValue(self->inst->getTopicsByDoc(doc->doc)), nullptr);
			}
		}
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_save(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	const char* filename;
	size_t full = 1;
	static const char* kwlist[] = { "filename", "full", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|n", (char**)kwlist, &filename, &full)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		ofstream str{ filename, ios_base::binary };
		if (!str) throw runtime_error{ "cannot open file '"s + filename + "'"s };
		self->inst->saveModel(str, !!full);
		Py_INCREF(Py_None);
		return Py_None;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_getDocs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Corpus_type, Py_BuildValue("(O)", self));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* LDA_getVocabs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Dictionary_type, Py_BuildValue("(Nn)", self, &self->inst->getVocabDict()));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}


DEFINE_GETTER(tomoto::ILDAModel, LDA, getK);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getAlpha);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getEta);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getPerplexity);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getLLPerWord);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getTermWeight);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getN);

static PyMethodDef LDA_methods[] =
{
	{ "add_doc", (PyCFunction)LDA_addDoc, METH_VARARGS | METH_KEYWORDS, LDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)LDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LDA_make_doc__doc__},
	{ "train", (PyCFunction)LDA_train, METH_VARARGS | METH_KEYWORDS, LDA_train__doc__},
	{ "get_topic_words", (PyCFunction)LDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_words__doc__},
	{ "get_topic_word_dist", (PyCFunction)LDA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_word_dist__doc__ },
	{ "infer", (PyCFunction)LDA_infer, METH_VARARGS | METH_KEYWORDS, LDA_infer__doc__ },
	{ "save", (PyCFunction)LDA_save, METH_VARARGS | METH_KEYWORDS, LDA_save__doc__},
	{ "load", (PyCFunction)LDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__},
	{ nullptr }
};

static PyGetSetDef LDA_getseters[] = {
	{ (char*)"tw", (getter)LDA_getTermWeight, nullptr, (char*)"term weighting scheme", NULL },
	{ (char*)"perplexity", (getter)LDA_getPerplexity, nullptr, (char*)"get perplexity of the model", NULL },
	{ (char*)"ll_per_word", (getter)LDA_getLLPerWord, nullptr, (char*)"get log likelihood per-word of the model", NULL },
	{ (char*)"k", (getter)LDA_getK, nullptr, (char*)"K, the number of topics", NULL },
	{ (char*)"alpha", (getter)LDA_getAlpha, nullptr, (char*)"hyperparameter alpha", NULL },
	{ (char*)"eta", (getter)LDA_getEta, nullptr, (char*)"hyperparameter eta", NULL },
	{ (char*)"docs", (getter)LDA_getDocs, nullptr, (char*)"get a `list`-like interface of `tomotopy.Document` in the model instance", NULL },
	{ (char*)"vocabs", (getter)LDA_getVocabs, nullptr, (char*)"get the dictionary of vocabuluary in type `tomotopy.Dictionary`", NULL },
	{ (char*)"num_words", (getter)LDA_getN, nullptr, (char*)"get the number of total words", NULL },
	{ nullptr },
};

static PyTypeObject LDA_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.LDA",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	LDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	LDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	LDA_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)LDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


static int DMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "k", "alpha", "eta", "sigma", "alpha_epsilon", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnffffn", (char**)kwlist, &tw, &K, &alpha, &eta, &sigma, &alphaEpsilon, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IDMRModel::create((tomoto::TermWeight)tw, K, alpha, sigma, eta, alphaEpsilon, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* DMR_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->addDoc(py::makeIterToVector(iter), { string{metadata} });
		return py::buildPyValue(ret);
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}


static PyObject* DMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->makeDoc(py::makeIterToVector(iter), { string{metadata} });
		return PyObject_CallObject((PyObject*)&Document_type, Py_BuildValue("(Nnn)", self, ret.release(), 1));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* DMR_getMetadataDict(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Dictionary_type, Py_BuildValue("(Nn)", self, 
			&static_cast<tomoto::IDMRModel*>(self->inst)->getMetadataDict()));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* DMR_getLambda(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		PyObject* ret = PyList_New(inst->getK());
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			PyList_SetItem(ret, i, py::buildPyValue(inst->getLambdaByTopic(i)));
		}
		return ret;
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef DMR_methods[] =
{
	{ "add_doc", (PyCFunction)DMR_addDoc, METH_VARARGS | METH_KEYWORDS, DMR_add_doc__doc__ },
	{ "make_doc", (PyCFunction)DMR_makeDoc, METH_VARARGS | METH_KEYWORDS, DMR_make_doc__doc__ },
	{ "load", (PyCFunction)DMR_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ nullptr }
};

DEFINE_GETTER(tomoto::IDMRModel, DMR, getAlphaEps);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getSigma);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getF);

static PyGetSetDef DMR_getseters[] = {
	{ (char*)"f", (getter)DMR_getF, nullptr, (char*)"number of features", NULL },
	{ (char*)"sigma", (getter)DMR_getSigma, nullptr, (char*)"sigma", NULL },
	{ (char*)"alpha_epsilon", (getter)DMR_getAlphaEps, nullptr, (char*)"alpha epsilon", NULL },
	{ (char*)"metadata_dict", (getter)DMR_getMetadataDict, nullptr, (char*)"get the dictionary of metadata in type `tomotopy.Dictionary`", NULL },
	{ (char*)"lambdas", (getter)DMR_getLambda, nullptr, (char*)"lambda", NULL },
	{ nullptr },
};


static PyTypeObject DMR_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.DMR",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	DMR___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	DMR_methods,             /* tp_methods */
	0,						 /* tp_members */
	DMR_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)DMR_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static int HDP_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, gamma = 0.1;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "initial_k", "alpha", "eta", "gamma", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnfffn", (char**)kwlist, &tw, &K, &alpha, &eta, &gamma, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IHDPModel::create((tomoto::TermWeight)tw, K, alpha, eta, gamma, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* HDP_isLiveTopic(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IHDPModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < K" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->isLiveTopic(topicId));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef HDP_methods[] =
{
	{ "load", (PyCFunction)HDP_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "is_live_topic", (PyCFunction)HDP_isLiveTopic, METH_VARARGS | METH_KEYWORDS, HDP_is_live_topic__doc__ },
	{ nullptr }
};

DEFINE_GETTER(tomoto::IHDPModel, HDP, getGamma);
DEFINE_GETTER(tomoto::IHDPModel, HDP, getTotalTables);
DEFINE_GETTER(tomoto::IHDPModel, HDP, getLiveK);

static PyGetSetDef HDP_getseters[] = {
	{ (char*)"gamma", (getter)HDP_getGamma, nullptr, (char*)"gamma", NULL },
	{ (char*)"live_k", (getter)HDP_getLiveK, nullptr, (char*)"number of alive topics", NULL },
	{ (char*)"num_tables", (getter)HDP_getTotalTables, nullptr, (char*)"number of total tables", NULL },
	{ nullptr },
};

static PyTypeObject HDP_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.HDP",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	HDP___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	HDP_methods,             /* tp_methods */
	0,						 /* tp_members */
	HDP_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)HDP_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


static int MGLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1, KL = 1, T = 3;
	float alpha = 0.1, alphaL = 0.1, eta = 0.01, etaL = 0.01, alphaM = 0.1, alphaML = 0.1, gamma = 0.1;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "k_g", "k_l", "t", "alpha_g", "alpha_l", "alpha_mg", "alpha_ml", 
		"eta_g", "eta_l", "gamma", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnfffffffn", (char**)kwlist, &tw, &K, &KL, &T,
		&alpha, &alphaL, &alphaM, &alphaML, &eta, &etaL, &gamma, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IMGLDAModel::create((tomoto::TermWeight)tw, 
			K, KL, T, alpha, alphaL, alphaM, alphaML, eta, etaL, gamma, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* MGLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	const char* delimiter = ".";
	static const char* kwlist[] = { "words", "delimiter", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &delimiter)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->addDoc(py::makeIterToVector(iter), delimiter);
		return py::buildPyValue(ret);
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* MGLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	const char* delimiter = ".";
	static const char* kwlist[] = { "words", "delimiter", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &delimiter)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words is not iterable" };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->makeDoc(py::makeIterToVector(iter), delimiter);
		return PyObject_CallObject((PyObject*)&Document_type, Py_BuildValue("(Nnn)", self, ret.release(), 1));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* MGLDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (topicId >= inst->getK() + inst->getKL()) throw runtime_error{ "must topic_id < KG + KL" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* MGLDA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (topicId >= inst->getK() + inst->getKL()) throw runtime_error{ "must topic_id < KG + KL" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWidsByTopic(topicId));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef MGLDA_methods[] =
{
	{ "load", (PyCFunction)MGLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "add_doc", (PyCFunction)MGLDA_addDoc, METH_VARARGS | METH_KEYWORDS, MGLDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)MGLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, MGLDA_make_doc__doc__ },
	{ "get_topic_words", (PyCFunction)MGLDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, MGLDA_get_topic_words__doc__ },
	{ "get_topic_word_dist", (PyCFunction)MGLDA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, MGLDA_get_topic_word_dist__doc__ },
	{ nullptr }
};

DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getKL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getGamma);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaM);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaML);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getEtaL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getT);

static PyGetSetDef MGLDA_getseters[] = {
	{ (char*)"k_g", (getter)LDA_getK, nullptr, (char*)"K global, number of global topics", NULL },
	{ (char*)"k_l", (getter)MGLDA_getKL, nullptr, (char*)"K local, number of local topics", NULL },
	{ (char*)"gamma", (getter)MGLDA_getGamma, nullptr, (char*)"gamma", NULL },
	{ (char*)"t", (getter)MGLDA_getT, nullptr, (char*)"window size", NULL },
	{ (char*)"alpha_g", (getter)LDA_getAlpha, nullptr, (char*)"alpha global", NULL },
	{ (char*)"alpha_l", (getter)MGLDA_getAlphaL, nullptr, (char*)"alpha local", NULL },
	{ (char*)"alpha_mg", (getter)MGLDA_getAlphaM, nullptr, (char*)"alpha mixture global", NULL },
	{ (char*)"alpha_ml", (getter)MGLDA_getAlphaML, nullptr, (char*)"alpha mixture local", NULL },
	{ (char*)"eta_g", (getter)LDA_getEta, nullptr, (char*)"eta global", NULL },
	{ (char*)"eta_l", (getter)MGLDA_getEtaL, nullptr, (char*)"eta local", NULL },
	{ nullptr },
};

static PyTypeObject MGLDA_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.MGLDA",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	MGLDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	MGLDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	MGLDA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)MGLDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static int PA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1, K2 = 1;
	float alpha = 0.1, eta = 0.01;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "k1", "k2", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnffn", (char**)kwlist, &tw, 
		&K, &K2, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IPAModel::create((tomoto::TermWeight)tw, 
			K, K2, alpha, eta, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* PA_getSubTopicDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < K" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getSubTopicBySuperTopic(topicId));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef PA_methods[] =
{
	{ "load", (PyCFunction)PA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_sub_topic_dist", (PyCFunction)PA_getSubTopicDist, METH_VARARGS | METH_KEYWORDS, PA_get_sub_topic_dist__doc__ },
	{ nullptr }
};

DEFINE_GETTER(tomoto::IPAModel, PA, getK2);

static PyGetSetDef PA_getseters[] = {
	{ (char*)"k1", (getter)LDA_getK, nullptr, (char*)"number of super topics", NULL },
	{ (char*)"k2", (getter)PA_getK2, nullptr, (char*)"number of sub topics", NULL },
	{ nullptr },
};

static PyTypeObject PA_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.PA",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	PA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	PA_methods,             /* tp_methods */
	0,						 /* tp_members */
	PA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static int HPA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0;
	size_t K = 1, K2 = 1;
	float alpha = 0.1, eta = 0.01;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "k1", "k2", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnffn", (char**)kwlist, &tw, 
		&K, &K2, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IHPAModel::create((tomoto::TermWeight)tw, 
			false, K, K2, alpha, eta, tomoto::RANDGEN{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* HPA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IHPAModel*>(self->inst);
		if (topicId > inst->getK() + inst->getK2()) throw runtime_error{ "must topic_id < 1 + K1 + K2" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* HPA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IHPAModel*>(self->inst);
		if (topicId > inst->getK() + inst->getK2()) throw runtime_error{ "must topic_id < 1 + K1 + K2" };
		if (!self->isPrepared)
		{
			inst->prepare();
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWidsByTopic(topicId));
	}
	catch (const bad_exception& e)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyMethodDef HPA_methods[] =
{
	{ "load", (PyCFunction)HPA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_topic_words", (PyCFunction)HPA_getTopicWords, METH_VARARGS | METH_KEYWORDS, HPA_get_topic_words__doc__ },
	{ "get_topic_word_dist", (PyCFunction)HPA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, HPA_get_topic_word_dist__doc__ },
	{ nullptr }
};

static PyGetSetDef HPA_getseters[] = {
	{ nullptr },
};

static PyTypeObject HPA_type = {
	PyVarObject_HEAD_INIT(NULL, 0)
	"tomotopy.HPA",             /* tp_name */
	sizeof(TopicModelObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)TopicModelObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	0,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,                         /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,                         /* tp_getattro */
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	HPA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	HPA_methods,             /* tp_methods */
	0,						 /* tp_members */
	HPA_getseters,                         /* tp_getset */
	&PA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)HPA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

DEFINE_LOADER(LDA, LDA_type);
DEFINE_LOADER(DMR, DMR_type);
DEFINE_LOADER(HDP, HDP_type);
DEFINE_LOADER(MGLDA, MGLDA_type);
DEFINE_LOADER(PA, PA_type);
DEFINE_LOADER(HPA, HPA_type);

PyMODINIT_FUNC MODULE_NAME()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"tomotopy",
		"Tomoto Module for Python",
		-1,
		nullptr,
	};

	if (PyType_Ready(&Document_type) < 0) return nullptr;
	if (PyType_Ready(&Corpus_type) < 0) return nullptr;
	if (PyType_Ready(&Dictionary_type) < 0) return nullptr;
	if (PyType_Ready(&LDA_type) < 0) return nullptr;
	if (PyType_Ready(&DMR_type) < 0) return nullptr;
	if (PyType_Ready(&HDP_type) < 0) return nullptr;
	if (PyType_Ready(&MGLDA_type) < 0) return nullptr;
	if (PyType_Ready(&PA_type) < 0) return nullptr;
	if (PyType_Ready(&HPA_type) < 0) return nullptr;

	gModule = PyModule_Create(&mod);
	if (!gModule) return nullptr;


#ifdef __AVX2__
	PyModule_AddStringConstant(gModule, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(gModule, "isa", "avx");
#elif defined(__SSE2__)
	PyModule_AddStringConstant(gModule, "isa", "sse2");
#else
	PyModule_AddStringConstant(gModule, "isa", "none");
#endif

	Py_INCREF(&Document_type);
	PyModule_AddObject(gModule, "Document", (PyObject*)&Document_type);
	Py_INCREF(&Corpus_type);
	PyModule_AddObject(gModule, "_Corpus", (PyObject*)&Corpus_type);
	Py_INCREF(&Dictionary_type);
	PyModule_AddObject(gModule, "Dictionary", (PyObject*)&Dictionary_type);
	Py_INCREF(&LDA_type);
	PyModule_AddObject(gModule, "LDA", (PyObject*)&LDA_type);
	Py_INCREF(&DMR_type);
	PyModule_AddObject(gModule, "DMR", (PyObject*)&DMR_type);
	Py_INCREF(&HDP_type);
	PyModule_AddObject(gModule, "HDP", (PyObject*)&HDP_type);
	Py_INCREF(&MGLDA_type);
	PyModule_AddObject(gModule, "MGLDA", (PyObject*)&MGLDA_type);
	Py_INCREF(&PA_type);
	PyModule_AddObject(gModule, "PA", (PyObject*)&PA_type);
	Py_INCREF(&HPA_type);
	PyModule_AddObject(gModule, "HPA", (PyObject*)&HPA_type);
	return gModule;
}
