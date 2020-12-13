#include "../TopicModel/HPA.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int HPA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1, K2 = 1;
	float alpha = 0.1f, eta = 0.01f;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k1", "k2", "alpha", "eta", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&K, &K2, &alpha, &eta, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IHPAModel::create((tomoto::TermWeight)tw, 
			false, K, K2, alpha, eta, seed);
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, K, K2, alpha, eta, seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
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
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
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
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getWidsByTopic(topicId));
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
DEFINE_LOADER(HPA, HPA_type);

PyObject* LDA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs);

static PyObject* HPA_getAlpha(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IHPAModel*>(self->inst);
		npy_intp shapes[1] = { (npy_intp)inst->getK() + 1 };
		PyObject* ret = PyArray_EMPTY(1, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i <= inst->getK(); ++i)
		{
			*(float*)PyArray_GETPTR1((PyArrayObject*)ret, i) = inst->getAlpha(i);
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

static PyObject* HPA_getSubalpha(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IHPAModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getK2() + 1 };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getSubAlpha(i);
			memcpy(PyArray_GETPTR2((PyArrayObject*)ret, i, 0), l.data(), sizeof(float) * l.size());
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

static PyMethodDef HPA_methods[] =
{
	{ "load", (PyCFunction)HPA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_topic_words", (PyCFunction)HPA_getTopicWords, METH_VARARGS | METH_KEYWORDS, HPA_get_topic_words__doc__ },
	{ "get_topic_word_dist", (PyCFunction)HPA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, HPA_get_topic_word_dist__doc__ },
	{ "infer", (PyCFunction)LDA_infer, METH_VARARGS | METH_KEYWORDS, LDA_infer__doc__ },
	{ nullptr }
};

static PyGetSetDef HPA_getseters[] = {
	{ (char*)"alpha", (getter)HPA_getAlpha, nullptr, HPA_alpha__doc__, nullptr },
	{ (char*)"subalpha", (getter)HPA_getSubalpha, nullptr, HPA_subalpha__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject HPA_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.HPAModel",             /* tp_name */
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
} };