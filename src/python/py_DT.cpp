#include "../TopicModel/DT.h"

#include "module.h"

using namespace std;

static int DT_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1, T = 1;
	float alphaVar = 0.1, etaVar = 0.1, phiVar = 0.1;
	float lrA = 0.01, lrB = 0.1, lrC = 0.55;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "t",
		"alpha_var", "eta_var", "phi_var", "lr_a", "lr_b", "lr_c",
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnffffffnOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &K, &T,
		&alphaVar, &etaVar, &phiVar, &lrA, &lrB, &lrC,
		&seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		tomoto::ITopicModel* inst = tomoto::IDTModel::create((tomoto::TermWeight)tw, K, T,
			alphaVar, etaVar, phiVar, lrA, lrB, lrC,
			0, tomoto::RandGen{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;

		if (objCorpus)
		{
			py::UniqueObj feeder = PyObject_GetAttrString(objCorpus, corpus_feeder_name),
				param = Py_BuildValue("(OO)", self, objTransform ? objTransform : Py_None);
			py::UniqueObj ret = PyObject_CallObject(feeder, param);
			if (!ret) return -1;
		}
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* DT_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	size_t timepoint = 0;
	static const char* kwlist[] = { "words", "timepoint", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|n", (char**)kwlist, &argWords, &timepoint)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto ret = inst->addDoc(py::makeIterToVector<string>(iter), timepoint);
		return py::buildPyValue(ret);
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

static PyObject* DT_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr;
	const char* argRaw = nullptr;
	size_t timepoint = 0;
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", "timepoint", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOOn", (char**)kwlist,
		&argWords, &argRaw, &argStartPos, &argLength, &timepoint)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;

		py::UniqueObj iter = PyObject_GetIter(argWords);
		vector<tomoto::Vid> words = py::makeIterToVector<tomoto::Vid>(iter);
		iter = PyObject_GetIter(argStartPos);
		vector<uint32_t> startPos = py::makeIterToVector<uint32_t>(iter);
		iter = PyObject_GetIter(argLength);
		vector<uint16_t> length = py::makeIterToVector<uint16_t>(iter);
		char2Byte(raw, startPos, length);
		auto ret = inst->addDoc(raw, words, startPos, length, timepoint);
		return py::buildPyValue(ret);
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

static PyObject* DT_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	size_t timepoint = 0;
	static const char* kwlist[] = { "words", "timepoint", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|n", (char**)kwlist, &argWords, &timepoint)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter), timepoint);
		py::UniqueObj args = Py_BuildValue("(Onn)", self, ret.release(), 1);
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

static PyObject* DT_getAlpha(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t timepoint;
	static const char* kwlist[] = { "timepoint", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &timepoint)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}

		vector<float> alphas;
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			alphas.emplace_back(inst->getAlpha(i, timepoint));
		}
		return py::buildPyValue(alphas);
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

static PyObject* DT_getPhi(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t timepoint, topicId;
	static const char* kwlist[] = { "timepoint", "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nn", (char**)kwlist, &timepoint, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}

		return py::buildPyValue(inst->getPhi(topicId, timepoint));
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

static PyObject* DT_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, timepoint, topN = 10;
	static const char* kwlist[] = { "topic_id", "timepoint", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nn|n", (char**)kwlist, &topicId, &timepoint, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k" };
		if (timepoint >= inst->getT()) throw runtime_error{ "must topic_id < t" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId + inst->getK() * timepoint, topN));
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

static PyObject* DT_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, timepoint;
	static const char* kwlist[] = { "topic_id", "timepoint", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "nn", (char**)kwlist, &topicId, &timepoint)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDTModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k" };
		if (timepoint >= inst->getT()) throw runtime_error{ "must topic_id < t" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getWidsByTopic(topicId + inst->getK() * timepoint));
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


DEFINE_LOADER(DT, DT_type);

static PyMethodDef DT_methods[] =
{
	{ "add_doc", (PyCFunction)DT_addDoc, METH_VARARGS | METH_KEYWORDS, DT_add_doc__doc__ },
	{ "_add_doc", (PyCFunction)DT_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
	{ "make_doc", (PyCFunction)DT_makeDoc, METH_VARARGS | METH_KEYWORDS, DT_make_doc__doc__ },
	{ "get_alpha", (PyCFunction)DT_getAlpha, METH_VARARGS | METH_KEYWORDS, DT_get_alpha__doc__ },
	{ "get_phi", (PyCFunction)DT_getPhi, METH_VARARGS | METH_KEYWORDS, DT_get_phi__doc__ },
	{ "get_topic_words", (PyCFunction)DT_getTopicWords, METH_VARARGS | METH_KEYWORDS, DT_get_topic_words__doc__ },
	{ "get_topic_word_dist", (PyCFunction)DT_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, DT_get_topic_word_dist__doc__ },
	{ "load", (PyCFunction)DT_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ nullptr }
};

DEFINE_GETTER(tomoto::IDTModel, DT, getShapeA);
DEFINE_GETTER(tomoto::IDTModel, DT, getShapeB);
DEFINE_GETTER(tomoto::IDTModel, DT, getShapeC);

DEFINE_SETTER_CHECKED_FLOAT(tomoto::IDTModel, DT, setShapeA, value > 0);
DEFINE_SETTER_CHECKED_FLOAT(tomoto::IDTModel, DT, setShapeB, value >= 0);
DEFINE_SETTER_CHECKED_FLOAT(tomoto::IDTModel, DT, setShapeC, 0.5 < value && value <= 1);

static PyGetSetDef DT_getseters[] = {
	{ (char*)"lr_a", (getter)DT_getShapeA, (setter)DT_setShapeA, DT_lr_a__doc__, nullptr },
	{ (char*)"lr_b", (getter)DT_getShapeB, (setter)DT_setShapeB, DT_lr_b__doc__, nullptr },
	{ (char*)"lr_c", (getter)DT_getShapeC, (setter)DT_setShapeC, DT_lr_c__doc__, nullptr },
	{ nullptr },
};

PyTypeObject DT_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.DTModel",             /* tp_name */
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
	DT___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	DT_methods,             /* tp_methods */
	0,						 /* tp_members */
	DT_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)DT_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};

PyObject* Document_eta(DocumentObject* self, void* closure)
{
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::one>*>(self->doc);
			if (doc) return py::buildPyValue(doc->eta.array().data(), doc->eta.array().data() + doc->eta.array().size());
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::idf>*>(self->doc);
			if (doc) return py::buildPyValue(doc->eta.array().data(), doc->eta.array().data() + doc->eta.array().size());
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::pmi>*>(self->doc);
			if (doc) return py::buildPyValue(doc->eta.array().data(), doc->eta.array().data() + doc->eta.array().size());
		} while (0);
		throw runtime_error{ "doc doesn't has 'eta' field!" };
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

PyObject* Document_timepoint(DocumentObject* self, void* closure)
{
	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::one>*>(self->doc);
			if (doc) return py::buildPyValue(doc->timepoint);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::idf>*>(self->doc);
			if (doc) return py::buildPyValue(doc->timepoint);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDTM<tomoto::TermWeight::pmi>*>(self->doc);
			if (doc) return py::buildPyValue(doc->timepoint);
		} while (0);
		throw runtime_error{ "doc doesn't has 'timepoint' field!" };
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
