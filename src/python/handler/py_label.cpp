#include "module.h"
#include "label.h"
#include "utils.h"
#include "label_docs.h"

using namespace std;

const string& CandWordIterator::operator*() const
{
	auto& v = co->tm ? co->tm->inst->getVocabDict() : *co->corpus->vocab->vocabs;
	return v.toWord(co->cand.w[idx]);
}

int CandidateObject::init(CandidateObject *self, PyObject *args, PyObject *kwargs)
{
	PyObject* words;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist,
		&words)) return -1;
	return py::handleExc([&]()
	{
		self->tm = nullptr;
		self->corpus = nullptr;
		new(&self->cand) tomoto::label::Candidate{};
		return 0;
	});
}

void CandidateObject::dealloc(CandidateObject* self)
{
	Py_XDECREF(self->tm);
	Py_XDECREF(self->corpus);
	self->cand.~Candidate();
	Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* CandidateObject::repr(CandidateObject* self)
{
	string ret = "tomotopy.label.Candidate(words=[";
	for (auto& w : *self)
	{
		ret.push_back('"');
		ret += w;
		ret.push_back('"');
		ret.push_back(',');
	}
	ret.back() = ']';
	ret += ", name=\"";
	ret += self->cand.name;
	ret += "\", score=";
	ret += to_string(self->cand.score);
	ret.push_back(')');
	return py::buildPyValue(ret);
}

static PyObject* Candidate_getWords(CandidateObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		return py::buildPyValue(self->begin(), self->end());
	});
}

static PyObject* Candidate_getName(CandidateObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		return py::buildPyValue(self->cand.name);
	});
}

int Candidate_setName(CandidateObject* self, PyObject* val, void* closure)
{
	return py::handleExc([&]()
	{
		if (!PyUnicode_Check(val)) throw runtime_error{ "`name` must be `str` type." };
		self->cand.name = PyUnicode_AsUTF8(val);
		return 0;
	});
}

static PyObject* Candidate_getScore(CandidateObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		return py::buildPyValue(self->cand.score);
	});
}

static PyObject* Candidate_getCf(CandidateObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		return py::buildPyValue(self->cand.cf);
	});
}

static PyObject* Candidate_getDf(CandidateObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		return py::buildPyValue(self->cand.df);
	});
}

static PyGetSetDef Candidate_getseters[] = {
	{ (char*)"words", (getter)Candidate_getWords, nullptr, Candidate_words__doc__, nullptr },
	{ (char*)"name", (getter)Candidate_getName, (setter)Candidate_setName, Candidate_name__doc__, nullptr },
	{ (char*)"score", (getter)Candidate_getScore, nullptr, Candidate_score__doc__, nullptr },
	{ (char*)"cf", (getter)Candidate_getCf, nullptr, Candidate_cf__doc__, nullptr },
	{ (char*)"df", (getter)Candidate_getDf, nullptr, Candidate_df__doc__, nullptr },
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
	(reprfunc)CandidateObject::repr, /* tp_repr */
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
	Candidate___init____doc__,           /* tp_doc */
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
		return py::handleExc([&]()
		{
			auto cands = self->inst->extract(tm->inst);
			PyObject* ret = PyList_New(0);
			for (auto& c : cands)
			{
				PyObject* item = PyObject_CallObject((PyObject*)&Candidate_type, nullptr);
				((CandidateObject*)item)->tm = tm;
				Py_INCREF(tm);
				((CandidateObject*)item)->cand = move(c);
				PyList_Append(ret, item);
			}
			return ret;
		});
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
		return py::handleExc([&]()
		{
			return py::buildPyValue(self->inst->getLabels(k, topN));
		});
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
	size_t minCf = 10, minDf = 5, minLen = 1, maxLen = 5, maxCand = 5000, normalized = 0;
	static const char* kwlist[] = { "min_cf", "min_df", "min_len", "max_len", "max_cand", "normalized", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnp", (char**)kwlist,
		&minCf, &minDf, &minLen, &maxLen, &maxCand, &normalized)) return -1;
	return py::handleExc([&]()
	{
		self->inst = new tomoto::label::PMIExtractor{ minCf, minDf, minLen, maxLen, maxCand, !!normalized };
		return 0;
	});
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
	size_t minDf = 5, windowSize = -1, numWorkers = 0;
	float smoothing = 1e-2f, mu = 0.25f;
	static const char* kwlist[] = { "topic_model", "cands", "min_df", "smoothing", "mu", "window_size", "workers", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|nffnn", (char**)kwlist,
		&tm, &cands, &minDf, &smoothing, &mu, &windowSize, &numWorkers)) return -1;
	return py::handleExc([&]()
	{
		self->tm = tm;
		self->inst = nullptr;
		Py_INCREF(tm);
		py::UniqueObj iter{ PyObject_GetIter(cands) };
		if (!iter) throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
		vector<tomoto::label::Candidate*> pcands;
		{
			py::UniqueObj item;
			while ((item = py::UniqueObj{ PyIter_Next(iter) }))
			{
				if (!PyObject_TypeCheck(item, &Candidate_type))
				{
					throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
				}
				pcands.emplace_back(&((CandidateObject*)item.get())->cand);
			}
		}
		auto deref = [](tomoto::label::Candidate* p)->tomoto::label::Candidate& { return *p; };
		self->inst = new tomoto::label::FoRelevance{
			tm->inst,
			tomoto::makeTransformIter(pcands.begin(), deref),
			tomoto::makeTransformIter(pcands.end(), deref),
			minDf, smoothing, 0, mu, windowSize, numWorkers
		};
		return 0;
	});
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


void addLabelTypes(PyObject* mModule)
{
	if (PyType_Ready(&Candidate_type) < 0) return;
	Py_INCREF(&Candidate_type);
	PyModule_AddObject(mModule, "_LabelCandidate", (PyObject*)&Candidate_type);
	if (PyType_Ready(&PMIExtractor_type) < 0) return;
	Py_INCREF(&PMIExtractor_type);
	PyModule_AddObject(mModule, "_LabelPMIExtractor", (PyObject*)&PMIExtractor_type);
	if (PyType_Ready(&FoRelevance_type) < 0) return;
	Py_INCREF(&FoRelevance_type);
	PyModule_AddObject(mModule, "_LabelFoRelevance", (PyObject*)&FoRelevance_type);
}
