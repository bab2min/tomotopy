#include "../TopicModel/PLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType PLDA_misc_args(TopicModelObject* self, const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["labels"] = getValueFromMiscDefault<vector<string>>("labels", o, "`PLDAModel` requires a `labels` value in `Iterable[str]` type.");
	return ret;
}

static int PLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::PLDAArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "latent_topics", "topics_per_label", "alpha", "eta",
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnOfOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.numLatentTopics, &margs.numTopicsPerLabel, &objAlpha, &margs.eta, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::IPLDAModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop,
			margs.numLatentTopics, margs.numTopicsPerLabel, margs.alpha, margs.eta, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* PLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr;
	size_t ignoreEmptyWords = 1;
	static const char* kwlist[] = { "words", "labels", "ignore_empty_words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", (char**)kwlist, &argWords, &argLabels, &ignoreEmptyWords)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (self->isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IPLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);

		if (argLabels)
		{
			if (PyUnicode_Check(argLabels))
			{
				if (PyErr_WarnEx(PyExc_RuntimeWarning, "`labels` should be an iterable of str.", 1)) return nullptr;
			}
			raw.misc["labels"] = py::toCpp<vector<string>>(argLabels, "`labels` must be an iterable of str.");
		}
		try
		{
			auto ret = inst->addDoc(raw);
			return py::buildPyValue(ret);
		}
		catch (const tomoto::exc::EmptyWordArgument&)
		{
			if (ignoreEmptyWords)
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
			else
			{
				throw;
			}
		}
	});
}

static DocumentObject* PLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr;
	static const char* kwlist[] = { "words", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argLabels)) return nullptr;
	return py::handleExc([&]() -> DocumentObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (!self->isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
		auto* inst = static_cast<tomoto::IPLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);

		if (argLabels)
		{
			if (PyUnicode_Check(argLabels))
			{
				if (PyErr_WarnEx(PyExc_RuntimeWarning, "`labels` should be an iterable of str.", 1)) return nullptr;
			}
			raw.misc["labels"] = py::toCpp<vector<string>>(argLabels, "`labels` must be an iterable of str.");
		}
		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
		return ret;
	});
}

static VocabObject* PLDA_getTopicLabelDict(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&static_cast<tomoto::IPLDAModel*>(self->inst)->getTopicLabelDict();
		ret->size = -1;
		return ret;
	});
}


DEFINE_LOADER(PLDA, PLDA_type);
DEFINE_GETTER(tomoto::IPLDAModel, PLDA, getNumLatentTopics);
DEFINE_GETTER(tomoto::IPLDAModel, PLDA, getNumTopicsPerLabel);

PyObject* LDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs);

static PyMethodDef PLDA_methods[] =
{
	{ "add_doc", (PyCFunction)PLDA_addDoc, METH_VARARGS | METH_KEYWORDS, LLDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)PLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LLDA_make_doc__doc__ },
	{ "load", (PyCFunction)PLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)PLDA_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "get_topic_words", (PyCFunction)LDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, PLDA_get_topic_words__doc__},
	{ nullptr }
};

static PyGetSetDef PLDA_getseters[] = {
	{ (char*)"topic_label_dict", (getter)PLDA_getTopicLabelDict, nullptr, PLDA_topic_label_dict__doc__, nullptr },
	{ (char*)"latent_topics", (getter)PLDA_getNumLatentTopics, nullptr, PLDA_latent_topics__doc__, nullptr },
	{ (char*)"topics_per_label", (getter)PLDA_getNumTopicsPerLabel, nullptr, PLDA_topics_per_label__doc__, nullptr },
	{ nullptr },
};


TopicModelTypeObject PLDA_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.PLDAModel",             /* tp_name */
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
	PLDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	PLDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	PLDA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PLDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
}, PLDA_misc_args };
