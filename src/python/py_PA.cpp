#include "../TopicModel/PA.h"

#include "module.h"

using namespace std;

static int PA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, rmTop = 0;
	size_t K = 1, K2 = 1;
	float alpha = 0.1, eta = 0.01;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "rm_top", "k1", "k2", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnffn", (char**)kwlist, &tw, &minCnt, &rmTop,
		&K, &K2, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IPAModel::create((tomoto::TermWeight)tw, 
			K, K2, alpha, eta, tomoto::RandGen{ seed });
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->removeTopWord = rmTop;
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
	static const char* kwlist[] = { "super_topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k1" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getSubTopicBySuperTopic(topicId));
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

static PyObject* PA_getSubTopics(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "super_topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k1" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getSubTopicBySuperTopicSorted(topicId, topN));

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

static PyObject* PA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "sub_topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK2()) throw runtime_error{ "must topic_id < k2" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
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

static PyObject* PA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "sub_topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK2()) throw runtime_error{ "must topic_id < k2" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
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

DEFINE_GETTER(tomoto::IPAModel, PA, getK2);
DEFINE_DOCUMENT_GETTER(tomoto::DocumentPA, Z2, Z2s);
DEFINE_LOADER(PA, PA_type);

static PyMethodDef PA_methods[] =
{
	{ "load", (PyCFunction)PA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_sub_topic_dist", (PyCFunction)PA_getSubTopicDist, METH_VARARGS | METH_KEYWORDS, PA_get_sub_topic_dist__doc__ },
	{ "get_sub_topics", (PyCFunction)PA_getSubTopics, METH_VARARGS | METH_KEYWORDS, PA_get_sub_topics__doc__ },
	{ "get_topic_words", (PyCFunction)PA_getTopicWords, METH_VARARGS | METH_KEYWORDS, PA_get_topic_words__doc__},
	{ "get_topic_word_dist", (PyCFunction)PA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, PA_get_topic_word_dist__doc__ },
	{ nullptr }
};


static PyGetSetDef PA_getseters[] = {
	{ (char*)"k1", (getter)LDA_getK, nullptr, PA_k1__doc__, nullptr },
	{ (char*)"k2", (getter)PA_getK2, nullptr, PA_k2__doc__, nullptr },
	{ nullptr },
};

PyTypeObject PA_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.PAModel",             /* tp_name */
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
