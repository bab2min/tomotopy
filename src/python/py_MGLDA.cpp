#include "../TopicModel/MGLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType MGLDA_misc_args(TopicModelObject* self, const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["delimiter"] = getValueFromMiscDefault<string>("delimiter", o, "`MGLDAModel` requires a `delimiter` value in `str` type.", ".");
	return ret;
}

static int MGLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::MGLDAArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k_g", "k_l", "t", "alpha_g", "alpha_l", "alpha_mg", "alpha_ml",
		"eta_g", "eta_l", "gamma", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnnfffffffOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &margs.kL, &margs.t, &margs.alpha[0], &margs.alphaL[0], &margs.alphaMG, &margs.alphaML, &margs.eta, &margs.etaL, &margs.gamma,
		&objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::IMGLDAModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop,
			margs.k, margs.kL, margs.t, margs.alpha[0], margs.alphaL[0],
			margs.alphaMG, margs.alphaML, margs.eta, margs.etaL, margs.gamma, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* MGLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	const char* delimiter = ".";
	size_t ignoreEmptyWords = 1;
	static const char* kwlist[] = { "words", "delimiter", "ignore_empty_words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sp", (char**)kwlist, &argWords, &delimiter, &ignoreEmptyWords)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (self->isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["delimiter"] = delimiter;
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

static DocumentObject* MGLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	const char* delimiter = ".";
	static const char* kwlist[] = { "words", "delimiter", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &delimiter)) return nullptr;
	return py::handleExc([&]() -> DocumentObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (!self->isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["delimiter"] = delimiter;
		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
		return ret;
	});
}

static PyObject* MGLDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (topicId >= inst->getK() + inst->getKL()) throw py::ValueError{ "must topic_id < KG + KL" };

		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
	});
}

static PyObject* MGLDA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, normalize = 1;
	static const char* kwlist[] = { "topic_id", "normalize", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|p", (char**)kwlist, &topicId, &normalize)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IMGLDAModel*>(self->inst);
		if (topicId >= inst->getK() + inst->getKL()) throw py::ValueError{ "must topic_id < KG + KL" };

		return py::buildPyValue(inst->getWidsByTopic(topicId, !!normalize));
	});
}

DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getKL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getGamma);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaM);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getAlphaML);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getEtaL);
DEFINE_GETTER(tomoto::IMGLDAModel, MGLDA, getT);

DEFINE_DOCUMENT_GETTER_REORDER(tomoto::DocumentMGLDA, windows, Vs);

DEFINE_LOADER(MGLDA, MGLDA_type);


static PyMethodDef MGLDA_methods[] =
{
	{ "load", (PyCFunction)MGLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)MGLDA_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "add_doc", (PyCFunction)MGLDA_addDoc, METH_VARARGS | METH_KEYWORDS, MGLDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)MGLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, MGLDA_make_doc__doc__ },
	{ "get_topic_words", (PyCFunction)MGLDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, MGLDA_get_topic_words__doc__ },
	{ "get_topic_word_dist", (PyCFunction)MGLDA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, MGLDA_get_topic_word_dist__doc__ },
	{ nullptr }
};

static PyGetSetDef MGLDA_getseters[] = {
	{ (char*)"k_g", (getter)LDA_getK, nullptr, MGLDA_k_g__doc__, nullptr },
	{ (char*)"k_l", (getter)MGLDA_getKL, nullptr, MGLDA_k_l__doc__, nullptr },
	{ (char*)"gamma", (getter)MGLDA_getGamma, nullptr, MGLDA_gamma__doc__, nullptr },
	{ (char*)"t", (getter)MGLDA_getT, nullptr, MGLDA_t__doc__, nullptr },
	{ (char*)"alpha_g", (getter)LDA_getAlpha, nullptr, MGLDA_alpha_g__doc__, nullptr },
	{ (char*)"alpha_l", (getter)MGLDA_getAlphaL, nullptr, MGLDA_alpha_l__doc__, nullptr },
	{ (char*)"alpha_mg", (getter)MGLDA_getAlphaM, nullptr, MGLDA_alpha_mg__doc__, nullptr },
	{ (char*)"alpha_ml", (getter)MGLDA_getAlphaML, nullptr, MGLDA_alpha_ml__doc__, nullptr },
	{ (char*)"eta_g", (getter)LDA_getEta, nullptr, MGLDA_eta_g__doc__, nullptr },
	{ (char*)"eta_l", (getter)MGLDA_getEtaL, nullptr, MGLDA_eta_l__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject MGLDA_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.MGLDAModel",             /* tp_name */
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
}, MGLDA_misc_args };
