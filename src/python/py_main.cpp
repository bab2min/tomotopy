#include "module.h"
#include "label.h"

#define TM_DMR
#define TM_HDP
#define TM_MGLDA
#define TM_PA
#define TM_HPA
#define TM_CT
#define TM_SLDA
#define TM_HLDA
#define TM_LLDA
#define TM_PLDA

using namespace std;

PyObject* gModule;


void char2Byte(const string& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	if (str.empty()) return;
	vector<size_t> charPos;
	auto it = str.begin(), end = str.end();
	for (; it != end; )
	{
		charPos.emplace_back(it - str.begin());
		uint8_t c = *it;
		if ((c & 0xF8) == 0xF0)
		{
			it += 4;
		}
		else if ((c & 0xF0) == 0xE0)
		{
			it += 3;
		}
		else if ((c & 0xE0) == 0xC0)
		{
			it += 2;
		}
		else if ((c & 0x80))
		{
			throw std::runtime_error{ "utf-8 decoding error" };
		}
		else it += 1;
	}
	charPos.emplace_back(str.size());

	for (size_t i = 0; i < startPos.size(); ++i)
	{
		size_t s = startPos[i], e = startPos[i] + length[i];
		startPos[i] = charPos[s];
		length[i] = charPos[e] - charPos[s];
	}
}


void TopicModelObject::dealloc(TopicModelObject* self)
{
	DEBUG_LOG("TopicModelObject Dealloc " << self);
	if (self->inst)
	{
		delete self->inst;
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}

void CorpusObject::dealloc(CorpusObject* self)
{
	DEBUG_LOG("CorpusObject Dealloc " << self->parentModel->ob_base.ob_type << ", " << self->parentModel->ob_base.ob_refcnt);
	Py_XDECREF(self->parentModel);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject * DocumentObject::repr(DocumentObject * self)
{
	string ret = "<tomotopy.Document with words=\"";
	for (size_t i = 0; i < self->doc->words.size(); ++i)
	{
		auto w = self->doc->wOrder.empty() ? self->doc->words[i] : self->doc->words[self->doc->wOrder[i]];
		ret += self->parentModel->inst->getVocabDict().toWord(w);
		ret.push_back(' ');
	}
	ret.pop_back();
	ret += "\">";
	return py::buildPyValue(ret);
}

void DocumentObject::dealloc(DocumentObject* self)
{
	DEBUG_LOG("DocumentObject Dealloc " << self->parentModel->ob_base.ob_type << ", " << self->parentModel->ob_base.ob_refcnt);
	Py_XDECREF(self->parentModel);
	if (self->owner)
	{
		delete self->doc;
	}
	Py_TYPE(self)->tp_free((PyObject*)self);
}

void DictionaryObject::dealloc(DictionaryObject* self)
{
	Py_XDECREF(self->parentModel);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

Py_ssize_t DictionaryObject::len(DictionaryObject* self)
{
	try
	{
		if (!self->dict) throw runtime_error{ "dict is null" };
		return self->dict->size();
	}
	catch (const bad_exception&)
	{
		return -1;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
}

PyObject* DictionaryObject::getitem(DictionaryObject* self, Py_ssize_t key)
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
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

PyObject* DictionaryObject::repr(DictionaryObject* self)
{
	py::UniqueObj args = Py_BuildValue("(O)", self);
	py::UniqueObj l = PyObject_CallObject((PyObject*)&PyList_Type, args);
	PyObject* r = PyObject_Repr(l);
	return r;
}

int DictionaryObject::init(DictionaryObject *self, PyObject *args, PyObject *kwargs)
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

static PySequenceMethods Dictionary_seq_methods = {
	(lenfunc)DictionaryObject::len,
	nullptr,
	nullptr,
	(ssizeargfunc)DictionaryObject::getitem,
};

PyTypeObject Dictionary_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.Dictionary",             /* tp_name */
	sizeof(DictionaryObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)DictionaryObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)DictionaryObject::repr, /* tp_repr */
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
		if (!self->parentModel->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(self->parentModel->inst->getTopicsByDocSorted(self->doc, topN));
	}
	catch (const bad_exception&)
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
		if (!self->parentModel->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(self->parentModel->inst->getTopicsByDoc(self->doc));
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyObject* Document_getWords(DocumentObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topN = 10;
	static const char* kwlist[] = { "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|n", (char**)kwlist, &topN)) return nullptr;
	try
	{
		if (!self->parentModel->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->parentModel->inst->getWordsByDocSorted(self->doc, topN));
	}
	catch (const bad_exception&)
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
#ifdef TM_PA
	{ "get_sub_topics", (PyCFunction)Document_getSubTopics, METH_VARARGS | METH_KEYWORDS, Document_get_sub_topics__doc__ },
	{ "get_sub_topic_dist", (PyCFunction)Document_getSubTopicDist, METH_NOARGS, Document_get_sub_topic_dist__doc__ },
#endif
	{ "get_words", (PyCFunction)Document_getWords, METH_VARARGS | METH_KEYWORDS, Document_get_words__doc__ },
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
		return buildPyValueReorder(self->doc->words, self->doc->wOrder);
	}
	catch (const bad_exception&)
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
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}


static PyObject* Document_Z(DocumentObject* self, void* closure)
{
	PyObject* ret;
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
#ifdef TM_HLDA
		ret = Document_HLDA_Z(self, closure);
		if (ret) return ret;
#endif
#ifdef TM_HDP
		ret = Document_HDP_Z(self, closure);
		if(ret) return ret;
#endif
		ret = Document_LDA_Z(self, closure);
		if(ret) return ret;
		throw runtime_error{ "doc doesn't has 'Zs' field!" };
	}
	catch (const bad_exception&)
	{
		return nullptr;
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return nullptr;
	}
}

static PyGetSetDef Document_getseters[] = {
	{ (char*)"words", (getter)Document_words, nullptr, Document_words__doc__, nullptr },
	{ (char*)"weight", (getter)Document_weight, nullptr, Document_weight__doc__, nullptr },
	{ (char*)"topics", (getter)Document_Z, nullptr, Document_topics__doc__, nullptr },
#ifdef TM_DMR
	{ (char*)"metadata", (getter)Document_metadata, nullptr, Document_metadata__doc__, nullptr },
#endif
#ifdef TM_PA
	{ (char*)"subtopics", (getter)Document_Z2, nullptr, Document_subtopics__doc__, nullptr },
#endif
#ifdef TM_MGLDA
	{ (char*)"windows", (getter)Document_windows, nullptr, Document_windows__doc__, nullptr },
#endif
#ifdef TM_CT
	{ (char*)"beta", (getter)Document_beta, nullptr, Document_beta__doc__, nullptr },
#endif
#ifdef TM_SLDA
	{ (char*)"vars", (getter)Document_y, nullptr, Document_vars__doc__, nullptr },
#endif
#ifdef TM_LLDA
	{ (char*)"labels", (getter)Document_labels, nullptr, Document_labels__doc__, nullptr },
#endif
	{ nullptr },
};


PyTypeObject Document_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.Document",             /* tp_name */
	sizeof(DocumentObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)DocumentObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)DocumentObject::repr, /* tp_repr */
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
	catch (const bad_exception&)
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
		py::UniqueObj args = Py_BuildValue("(Onn)", self->parentModel, key, 0);
		return PyObject_CallObject((PyObject*)&Document_type, args);
	}
	catch (const bad_exception&)
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


PyTypeObject Corpus_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy._Corpus",             /* tp_name */
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
	"",           /* tp_doc */
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

	gModule = PyModule_Create(&mod);
	if (!gModule) return nullptr;


	if (PyType_Ready(&Document_type) < 0) return nullptr;
	Py_INCREF(&Document_type);
	PyModule_AddObject(gModule, "Document", (PyObject*)&Document_type);
	if (PyType_Ready(&Corpus_type) < 0) return nullptr;
	Py_INCREF(&Corpus_type);
	PyModule_AddObject(gModule, "_Corpus", (PyObject*)&Corpus_type);
	if (PyType_Ready(&Dictionary_type) < 0) return nullptr;
	Py_INCREF(&Dictionary_type);
	PyModule_AddObject(gModule, "Dictionary", (PyObject*)&Dictionary_type);
	if (PyType_Ready(&LDA_type) < 0) return nullptr;
	Py_INCREF(&LDA_type);
	PyModule_AddObject(gModule, "LDAModel", (PyObject*)&LDA_type);

#ifdef TM_DMR
	if (PyType_Ready(&DMR_type) < 0) return nullptr;
	Py_INCREF(&DMR_type);
	PyModule_AddObject(gModule, "DMRModel", (PyObject*)&DMR_type);
#endif
#ifdef TM_HDP
	if (PyType_Ready(&HDP_type) < 0) return nullptr;
	Py_INCREF(&HDP_type);
	PyModule_AddObject(gModule, "HDPModel", (PyObject*)&HDP_type);
#endif
#ifdef TM_MGLDA
	if (PyType_Ready(&MGLDA_type) < 0) return nullptr;
	Py_INCREF(&MGLDA_type);
	PyModule_AddObject(gModule, "MGLDAModel", (PyObject*)&MGLDA_type);
#endif
#ifdef TM_PA
	if (PyType_Ready(&PA_type) < 0) return nullptr;
	Py_INCREF(&PA_type);
	PyModule_AddObject(gModule, "PAModel", (PyObject*)&PA_type);
#endif
#ifdef TM_HPA
	if (PyType_Ready(&HPA_type) < 0) return nullptr;
	Py_INCREF(&HPA_type);
	PyModule_AddObject(gModule, "HPAModel", (PyObject*)&HPA_type);
#endif
#ifdef TM_HLDA
	if (PyType_Ready(&HLDA_type) < 0) return nullptr;
	Py_INCREF(&HLDA_type);
	PyModule_AddObject(gModule, "HLDAModel", (PyObject*)&HLDA_type);
#endif
#ifdef TM_CT
	if (PyType_Ready(&CT_type) < 0) return nullptr;
	Py_INCREF(&CT_type);
	PyModule_AddObject(gModule, "CTModel", (PyObject*)&CT_type);
#endif
#ifdef TM_SLDA
	if (PyType_Ready(&SLDA_type) < 0) return nullptr;
	Py_INCREF(&SLDA_type);
	PyModule_AddObject(gModule, "SLDAModel", (PyObject*)&SLDA_type);
#endif
#ifdef TM_LLDA
	if (PyType_Ready(&LLDA_type) < 0) return nullptr;
	Py_INCREF(&LLDA_type);
	PyModule_AddObject(gModule, "LLDAModel", (PyObject*)&LLDA_type);
#endif
#ifdef TM_PLDA
	if (PyType_Ready(&PLDA_type) < 0) return nullptr;
	Py_INCREF(&PLDA_type);
	PyModule_AddObject(gModule, "PLDAModel", (PyObject*)&PLDA_type);
#endif

#ifdef __AVX2__
	PyModule_AddStringConstant(gModule, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(gModule, "isa", "avx");
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_WIN64)
	PyModule_AddStringConstant(gModule, "isa", "sse2");
#else
	PyModule_AddStringConstant(gModule, "isa", "none");
#endif
	PyObject* sModule = makeLabelModule();
	if (!sModule) return nullptr;
	Py_INCREF(sModule);
	PyModule_AddObject(gModule, "label", sModule);

	return gModule;
}
