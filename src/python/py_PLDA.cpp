#include "../TopicModel/PLDA.h"

#include "module.h"

using namespace std;

static int PLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t numLatentTopics = 0, numTopicsPerLabel = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "latent_topics", "topics_per_label", "alpha", "eta", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&numLatentTopics, &numTopicsPerLabel, &alpha, &eta, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		tomoto::ITopicModel* inst = tomoto::IPLDAModel::create((tomoto::TermWeight)tw, 
			numLatentTopics, numTopicsPerLabel, alpha, eta, tomoto::RandGen{ seed });
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
			if(!ret) return -1;
		}
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
		return -1;
	}
	return 0;
}

static PyObject* PLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr;
	static const char* kwlist[] = { "words", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argLabels)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IPLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		vector<string> labels;
		if(argLabels)
		{
			if (PyUnicode_Check(argLabels)) PRINT_WARN("[warn] 'labels' should be an iterable of str.");
			py::UniqueObj iter2;
			if (!(iter2 = PyObject_GetIter(argLabels)))
			{
				throw runtime_error{ "'labels' must be an iterable of str." };
			}
			labels = py::makeIterToVector<string>(iter2);
		}
		auto ret = inst->addDoc(py::makeIterToVector<string>(iter), labels);
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

static PyObject* PLDA_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr, *argLabels = nullptr;
	const char* argRaw = nullptr;
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOOO", (char**)kwlist,
		&argWords, &argRaw, &argStartPos, &argLength, &argLabels)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPLDAModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;

		py::UniqueObj iter = PyObject_GetIter(argWords);
		vector<tomoto::Vid> words = py::makeIterToVector<tomoto::Vid>(iter);
		iter = PyObject_GetIter(argStartPos);
		vector<uint32_t> startPos = py::makeIterToVector<uint32_t>(iter);
		iter = PyObject_GetIter(argLength);
		vector<uint16_t> length = py::makeIterToVector<uint16_t>(iter);
		char2Byte(raw, startPos, length);
		vector<string> labels;
		if (argLabels)
		{
			py::UniqueObj iter2;
			if (PyUnicode_Check(argLabels)) PRINT_WARN("[warn] 'labels' should be an iterable of str.");
			if (!(iter2 = PyObject_GetIter(argLabels)))
			{
				throw runtime_error{ "'labels' must be an iterable of str." };
			}
			labels = py::makeIterToVector<string>(iter2);
		}

		auto ret = inst->addDoc(raw, words, startPos, length, labels);
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

static PyObject* PLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr;
	static const char* kwlist[] = { "words", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argLabels)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		vector<string> labels;
		if (argLabels)
		{
			if (PyUnicode_Check(argLabels)) PRINT_WARN("[warn] 'labels' should be an iterable of str.");
			py::UniqueObj iter2;
			if (!(iter2 = PyObject_GetIter(argLabels)))
			{
				throw runtime_error{ "'labels' must be an iterable of str." };
			}
			labels = py::makeIterToVector<string>(iter2);
		}
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter), labels);
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

static PyObject* PLDA_getTopicLabelDict(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		py::UniqueObj args = Py_BuildValue("(On)", self,
			&static_cast<tomoto::IPLDAModel*>(self->inst)->getTopicLabelDict());
		return PyObject_CallObject((PyObject*)&Dictionary_type, args);
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


DEFINE_LOADER(PLDA, PLDA_type);
DEFINE_GETTER(tomoto::IPLDAModel, PLDA, getNumLatentTopics);
DEFINE_GETTER(tomoto::IPLDAModel, PLDA, getNumTopicsPerLabel);

PyObject* LDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs);

static PyMethodDef PLDA_methods[] =
{
	{ "add_doc", (PyCFunction)PLDA_addDoc, METH_VARARGS | METH_KEYWORDS, LLDA_add_doc__doc__ },
	{ "_add_doc", (PyCFunction)PLDA_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
	{ "make_doc", (PyCFunction)PLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LLDA_make_doc__doc__ },
	{ "load", (PyCFunction)PLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_topic_words", (PyCFunction)LDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, PLDA_get_topic_words__doc__},
	{ nullptr }
};

static PyGetSetDef PLDA_getseters[] = {
	{ (char*)"topic_label_dict", (getter)PLDA_getTopicLabelDict, nullptr, PLDA_topic_label_dict__doc__, nullptr },
	{ (char*)"latent_topics", (getter)PLDA_getNumLatentTopics, nullptr, PLDA_latent_topics__doc__, nullptr },
	{ (char*)"topics_per_label", (getter)PLDA_getNumTopicsPerLabel, nullptr, PLDA_topics_per_label__doc__, nullptr },
	{ nullptr },
};


PyTypeObject PLDA_type = {
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
};
