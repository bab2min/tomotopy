
#include "module.h"
#include "coherence.h"

using namespace std;

int CoherenceObject::init(CoherenceObject* self, PyObject* args, PyObject* kwargs)
{
	new (&self->model) tomoto::coherence::CoherenceModel;
	new (&self->cm) tomoto::coherence::AnyConfirmMeasurer;

	CorpusObject* corpus;
	PyObject* targets = nullptr;
	size_t windowSize = 0;
	double eps = 1e-12;
	double gamma = 1;
	ProbEstimation pe = ProbEstimation::none;
	Segmentation seg = Segmentation::none;
	ConfirmMeasure cm = ConfirmMeasure::none;
	IndirectMeasure im = IndirectMeasure::none;
	static const char* kwlist[] = { "corpus", "pe", "seg", "cm", "im", "window_size", "eps", "gamma", "targets", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iiiinddO", (char**)kwlist,
		&corpus, &pe, &seg, &cm, &im, &windowSize, &eps, &gamma, &targets)) return -1;
	return py::handleExc([&]()
	{
		if (!PyObject_TypeCheck(corpus, &UtilsCorpus_type))
		{
			throw py::ValueError{ "`corpus` must be an instance of `tomotopy.utils.Corpus`." };
		}
		self->model.~CoherenceModel();
		new (&self->model) tomoto::coherence::CoherenceModel{ pe, windowSize };

		self->corpus = corpus;
		Py_INCREF(corpus);

		vector<tomoto::Vid> targetIds;
		py::foreach<string>(targets, [&](const string& w)
		{
			auto wid = corpus->getVocabDict().toWid(w);
			if (wid != tomoto::non_vocab_id) targetIds.emplace_back(wid);
		}, "`targets` must be an iterable of `str`.");

		self->model.insertTargets(targetIds.begin(), targetIds.end());

		for (size_t i = 0; i < CorpusObject::len(corpus); ++i)
		{
			auto* doc = corpus->getDoc(i);
			self->model.insertDoc(
				wordBegin(doc, corpus->isIndependent()),
				wordEnd(doc, corpus->isIndependent())
			);
		}

		self->seg = seg;
		self->cm = tomoto::coherence::AnyConfirmMeasurer::getInstance(cm, im, targetIds.begin(), targetIds.end(), eps, gamma);
		return 0;
	});
}

PyObject* CoherenceObject::repr(CoherenceObject* self)
{
	return py::buildPyValue(string{ });
}

void CoherenceObject::dealloc(CoherenceObject* self)
{
	self->model.~CoherenceModel();
	self->cm.~AnyConfirmMeasurer();
	Py_XDECREF(self->corpus);
	Py_TYPE(self)->tp_free((PyObject*)self);
}

PyObject* CoherenceObject::getScore(CoherenceObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject* words;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist,
		&words)) return nullptr;
	
	return py::handleExc([&]()
	{
		vector<tomoto::Vid> wordIds;
		py::foreach<string>(words, [&](const string& w)
		{
			auto wid = self->corpus->getVocabDict().toWid(w);
			if (wid != tomoto::non_vocab_id) wordIds.emplace_back(wid);
		}, "`words` must be an iterable of `str`.");

		switch (self->seg)
		{
		case Segmentation::one_one:
			return py::buildPyValue(self->model.template getScore<Segmentation::one_one>(self->cm, wordIds.begin(), wordIds.end()));
		case Segmentation::one_pre:
			return py::buildPyValue(self->model.template getScore<Segmentation::one_pre>(self->cm, wordIds.begin(), wordIds.end()));
		case Segmentation::one_suc:
			return py::buildPyValue(self->model.template getScore<Segmentation::one_suc>(self->cm, wordIds.begin(), wordIds.end()));
		case Segmentation::one_all:
			return py::buildPyValue(self->model.template getScore<Segmentation::one_all>(self->cm, wordIds.begin(), wordIds.end()));
		case Segmentation::one_set:
			return py::buildPyValue(self->model.template getScore<Segmentation::one_set>(self->cm, wordIds.begin(), wordIds.end()));
		default:
			throw py::ValueError{ "invalid Segmentation `seg`" };
		}
	});
}

static PyMethodDef Coherence_methods[] =
{
	{ "get_score", (PyCFunction)CoherenceObject::getScore, METH_VARARGS | METH_KEYWORDS, "" },
	{ nullptr }
};


static PyGetSetDef Coherence_getseters[] = {
	//{ (char*)"words", (getter)CoherenceObject::getWords, nullptr, Document_words__doc__, nullptr },
	{ nullptr }
};

PyTypeObject Coherence_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy._Coherence",             /* tp_name */
	sizeof(CoherenceObject), /* tp_basicsize */
	0,                         /* tp_itemsize */
	(destructor)CoherenceObject::dealloc, /* tp_dealloc */
	0,                         /* tp_print */
	0,                         /* tp_getattr */
	0,                         /* tp_setattr */
	0,                         /* tp_reserved */
	(reprfunc)CoherenceObject::repr,                         /* tp_repr */
	0,                         /* tp_as_number */
	0,       /* tp_as_sequence */
	0,                         /* tp_as_mapping */
	0,                         /* tp_hash  */
	0,                         /* tp_call */
	0,                         /* tp_str */
	0,
	0,                         /* tp_setattro */
	0,                         /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
	"",           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,              /* tp_iter */
	0,                         /* tp_iternext */
	Coherence_methods,             /* tp_methods */
	0,						 /* tp_members */
	Coherence_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CoherenceObject::init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};


void addCoherenceTypes(PyObject* gModule)
{
	if (PyType_Ready(&Coherence_type) < 0) throw runtime_error{ "Coherence_type is not ready." };
	Py_INCREF(&Coherence_type);
	PyModule_AddObject(gModule, "_Coherence", (PyObject*)&Coherence_type);
}
