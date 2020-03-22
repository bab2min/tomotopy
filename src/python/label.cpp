#include "../Labeling/FoRelevance.h"

#include "module.h"
#include "label.h"
#include "label_docs.h"

using namespace std;

struct CandidateObject
{
	PyObject_HEAD;
	TopicModelObject* tm;
	tomoto::label::Candidate cand;

	static int init(CandidateObject *self, PyObject *args, PyObject *kwargs)
	{
		PyObject* tm;
		static const char* kwlist[] = { "tm", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
			&tm)) return -1;
		try
		{
			self->tm = (TopicModelObject*)tm;
			Py_INCREF(tm);
			new(&self->cand) tomoto::label::Candidate{};
		}
		catch (const exception& e)
		{
			PyErr_SetString(PyExc_Exception, e.what());
			return -1;
		}
		return 0;
	}

	static void dealloc(CandidateObject* self)
	{
		Py_XDECREF(self->tm);
		self->cand.~Candidate();
	}
};

static PyObject* Candidate_getWords(CandidateObject* self, void* closure)
{
	try
	{
		auto& v = self->tm->inst->getVocabDict();
		return py::buildPyValueTransform(self->cand.w.begin(), self->cand.w.end(), [&](tomoto::Vid w)
		{
			return v.toWord(w);
		});
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

static PyObject* Candidate_getName(CandidateObject* self, void* closure)
{
	try
	{
		return py::buildPyValue(self->cand.name);
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

int Candidate_setName(CandidateObject* self, PyObject* val, void* closure)
{
	try
	{
		if (!PyUnicode_Check(val)) throw runtime_error{ "`name` must be `str` type." };
		self->cand.name = PyUnicode_AsUTF8(val);
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
	return 0;
}

static PyObject* Candidate_getScore(CandidateObject* self, void* closure)
{
	try
	{
		return py::buildPyValue(self->cand.score);
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

static PyGetSetDef Candidate_getseters[] = {
	{ (char*)"words", (getter)Candidate_getWords, nullptr, Candidate_words__doc__, nullptr },
	{ (char*)"name", (getter)Candidate_getName, (setter)Candidate_setName, Candidate_name__doc__, nullptr },
	{ (char*)"score", (getter)Candidate_getScore, nullptr, Candidate_score__doc__, nullptr },
	{ nullptr },
};

PyTypeObject Candidate_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.label.Candidate",             /* tp_name */
	sizeof(CandidateObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CandidateObject::dealloc, /* tp_dealloc */
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
	"Candidate",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	0,             /* tp_methods */
	0,						 /* tp_members */
	Candidate_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CandidateObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

struct ExtractorObject
{
	PyObject_HEAD;
	tomoto::label::IExtractor* inst;

	static PyObject* extract(ExtractorObject* self, PyObject* args, PyObject *kwargs)
	{
		TopicModelObject* tm;
		static const char* kwlist[] = { "topic_model", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &tm)) return nullptr;
		try
		{
			auto cands = self->inst->extract(tm->inst);
			PyObject* ret = PyList_New(0);
			for (auto& c : cands)
			{
				py::UniqueObj param = Py_BuildValue("(O)", tm);
				PyObject* item = PyObject_CallObject((PyObject*)&Candidate_type, param);
				PyList_Append(ret, item);
				((CandidateObject*)item)->cand = c;
			}
			return ret;
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

	static void dealloc(ExtractorObject* self)
	{
		delete self->inst;
	}
};

static PyMethodDef Extractor_methods[] =
{
	{ "extract", (PyCFunction)ExtractorObject::extract, METH_VARARGS | METH_KEYWORDS, Extractor_extract__doc__ },
	{ nullptr }
};

struct LabelerObject
{
	PyObject_HEAD;
	tomoto::label::ILabeler* inst;
	TopicModelObject* tm;

	static PyObject* getTopicLabels(LabelerObject* self, PyObject* args, PyObject *kwargs)
	{
		size_t k, topN = 10;
		static const char* kwlist[] = { "k", "top_n", nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, 
			&k, &topN)) return nullptr;
		try
		{
			return py::buildPyValue(self->inst->getLabels(k, topN));
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

	static void dealloc(LabelerObject* self)
	{
		delete self->inst;
		Py_XDECREF(self->tm);
	}
};

static PyMethodDef Labeler_methods[] =
{
	{ "get_topic_labels", (PyCFunction)LabelerObject::getTopicLabels, METH_VARARGS | METH_KEYWORDS, Labeler_get_topic_labels__doc__ },
	{ nullptr }
};

static int PMIExtractor_init(ExtractorObject *self, PyObject *args, PyObject *kwargs)
{
	size_t minCf = 10, minDf = 5, maxLen = 5, maxCand = 5000;
	static const char* kwlist[] = { "min_cf", "min_df", "max_len", "max_cand", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnn", (char**)kwlist,
		&minCf, &minDf, &maxLen, &maxCand)) return -1;
	try
	{
		self->inst = new tomoto::label::PMIExtractor{ minCf, minDf, maxLen, maxCand };
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

PyTypeObject PMIExtractor_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.label.PMIExtractor",             /* tp_name */
	sizeof(ExtractorObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)ExtractorObject::dealloc, /* tp_dealloc */
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
	PMIExtractor___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	Extractor_methods,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PMIExtractor_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

static int FoRelevance_init(LabelerObject *self, PyObject *args, PyObject *kwargs)
{
	TopicModelObject* tm;
	PyObject* cands;
	size_t minDf = 5, numWorkers = 0;
	float smoothing = 1e-2f, mu = 0.25f;
	static const char* kwlist[] = { "topic_model", "cands", "min_df", "smoothing", "mu", "workers", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|nffn", (char**)kwlist,
		&tm, &cands, &minDf, &smoothing, &mu, &numWorkers)) return -1;
	try
	{
		self->tm = tm;
		self->inst = nullptr;
		Py_INCREF(tm);
		py::UniqueObj iter = PyObject_GetIter(cands);
		if (!iter)
		{
			throw runtime_error{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
		}
		vector<tomoto::label::Candidate*> pcands;
		{
			py::UniqueObj item;
			while ((item = PyIter_Next(iter)))
			{
				if(!PyObject_TypeCheck(item, &Candidate_type)) throw runtime_error{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
				pcands.emplace_back(&((CandidateObject*)item.get())->cand);
			}
		}
		auto deref = [](tomoto::label::Candidate* p) { return *p; };
		self->inst = new tomoto::label::FoRelevance{ 
			tm->inst, 
			tomoto::makeTransformIter(pcands.begin(), deref), 
			tomoto::makeTransformIter(pcands.end(), deref), 
			minDf, smoothing, 0, mu, numWorkers 
		};
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

PyTypeObject FoRelevance_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.label.FoRelevance",             /* tp_name */
	sizeof(LabelerObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)LabelerObject::dealloc, /* tp_dealloc */
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
	FoRelevance___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	Labeler_methods,             /* tp_methods */
	0,						 /* tp_members */
	0,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)FoRelevance_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


PyObject* makeLabelModule()
{
	static PyModuleDef mod =
	{
		PyModuleDef_HEAD_INIT,
		"tomotopy.label",
		"Auto labeling package for tomotopy",
		-1,
		nullptr,
	};

	PyObject* mModule = PyModule_Create(&mod);

	if (PyType_Ready(&Candidate_type) < 0) return nullptr;
	Py_INCREF(&Candidate_type);
	PyModule_AddObject(mModule, "Candidate", (PyObject*)&Candidate_type);
	if (PyType_Ready(&PMIExtractor_type) < 0) return nullptr;
	Py_INCREF(&PMIExtractor_type);
	PyModule_AddObject(mModule, "PMIExtractor", (PyObject*)&PMIExtractor_type);
	if (PyType_Ready(&FoRelevance_type) < 0) return nullptr;
	Py_INCREF(&FoRelevance_type);
	PyModule_AddObject(mModule, "FoRelevance", (PyObject*)&FoRelevance_type);
	return mModule;
}
