#include "../TopicModel/HLDA.h"

#include "module.h"

using namespace std;

static int HLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t depth = 2;
	float alpha = 0.1, eta = 0.01, gamma = 0.1;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "depth", "alpha", "eta", "gamma", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnfffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&depth, &alpha, &eta, &gamma, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		tomoto::ITopicModel* inst = tomoto::IHLDAModel::create((tomoto::TermWeight)tw, depth, alpha, eta, gamma, tomoto::RandGen{ seed });
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

#define DEFINE_HLDA_TOPIC_METH(NAME) \
static PyObject* HLDA_##NAME(TopicModelObject* self, PyObject* args, PyObject *kwargs)\
{\
	size_t topicId;\
	static const char* kwlist[] = { "topic_id", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;\
	try\
	{\
		if (!self->inst) throw runtime_error{ "inst is null" };\
		auto* inst = static_cast<tomoto::IHLDAModel*>(self->inst);\
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < K" };\
		if (!self->isPrepared) throw runtime_error{ "train() should be called first" };\
		return py::buildPyValue(inst->NAME(topicId));\
	}\
	catch (const bad_exception&)\
	{\
		return nullptr;\
	}\
	catch (const exception& e)\
	{\
		PyErr_SetString(PyExc_Exception, e.what());\
		return nullptr;\
	}\
}


PyObject* Document_HLDA_Z(DocumentObject* self, void* closure)
{
	do
	{
		auto* doc = dynamic_cast<const tomoto::DocumentHLDA<tomoto::TermWeight::one>*>(self->doc);
		if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](size_t x) { return doc->path[x]; });
	} while (0);
	do
	{
		auto* doc = dynamic_cast<const tomoto::DocumentHLDA<tomoto::TermWeight::idf>*>(self->doc);
		if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](size_t x) { return doc->path[x]; });
	} while (0);
	do
	{
		auto* doc = dynamic_cast<const tomoto::DocumentHLDA<tomoto::TermWeight::pmi>*>(self->doc);
		if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](size_t x) { return doc->path[x]; });
	} while (0);
	return nullptr;
}


DEFINE_GETTER(tomoto::IHLDAModel, HLDA, getGamma);
DEFINE_GETTER(tomoto::IHLDAModel, HLDA, getLevelDepth);
DEFINE_GETTER(tomoto::IHLDAModel, HLDA, getLiveK);

DEFINE_LOADER(HLDA, HLDA_type);

DEFINE_HLDA_TOPIC_METH(isLiveTopic);
DEFINE_HLDA_TOPIC_METH(getNumDocsOfTopic);
DEFINE_HLDA_TOPIC_METH(getLevelOfTopic);
DEFINE_HLDA_TOPIC_METH(getParentTopicId);
DEFINE_HLDA_TOPIC_METH(getChildTopicId);


static PyMethodDef HLDA_methods[] =
{
	{ "load", (PyCFunction)HLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "is_live_topic", (PyCFunction)HLDA_isLiveTopic, METH_VARARGS | METH_KEYWORDS, HLDA_is_live_topic__doc__ },
	{ "num_docs_of_topic", (PyCFunction)HLDA_getNumDocsOfTopic, METH_VARARGS | METH_KEYWORDS, HLDA_num_docs_of_topic__doc__ },
	{ "level", (PyCFunction)HLDA_getLevelOfTopic, METH_VARARGS | METH_KEYWORDS, HLDA_level__doc__ },
	{ "parent_topic", (PyCFunction)HLDA_getParentTopicId, METH_VARARGS | METH_KEYWORDS, HLDA_parent_topic__doc__ },
	{ "children_topics", (PyCFunction)HLDA_getChildTopicId, METH_VARARGS | METH_KEYWORDS, HLDA_children_topics__doc__ },
	{ nullptr }
};

static PyGetSetDef HLDA_getseters[] = {
	{ (char*)"gamma", (getter)HLDA_getGamma, nullptr, HLDA_gamma__doc__, nullptr },
	{ (char*)"live_k", (getter)HLDA_getLiveK, nullptr, HLDA_live_k__doc__, nullptr },
	{ (char*)"depth", (getter)HLDA_getLevelDepth, nullptr, HLDA_depth__doc__, nullptr },
	{ nullptr },
};

PyTypeObject HLDA_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.HLDAModel",             /* tp_name */
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
	HLDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	HLDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	HLDA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)HLDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};
