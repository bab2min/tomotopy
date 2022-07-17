#include "../TopicModel/HLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int HLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::HLDAArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "depth", "alpha", "eta", "gamma", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOffOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &objAlpha, &margs.eta, &margs.gamma, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `depth` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::IHLDAModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, margs.k, margs.alpha, margs.eta, margs.gamma, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

#define DEFINE_HLDA_TOPIC_METH(NAME) \
static PyObject* HLDA_##NAME(TopicModelObject* self, PyObject* args, PyObject *kwargs)\
{\
	size_t topicId;\
	static const char* kwlist[] = { "topic_id", nullptr };\
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;\
	return py::handleExc([&]()\
	{\
		if (!self->inst) throw py::RuntimeError{ "inst is null" };\
		auto* inst = static_cast<tomoto::IHLDAModel*>(self->inst);\
		if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < K" };\
		if (!self->isPrepared) throw py::RuntimeError{ "train() should be called first" };\
		return py::buildPyValue(inst->NAME(topicId));\
	});\
}


PyObject* Document_HLDA_Z(DocumentObject* self, void* closure)
{
	return docVisit<tomoto::DocumentHLDA>(self->getBoundDoc(), [](auto* doc)
	{
		return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](tomoto::Tid x) -> int16_t
		{ 
			if (x == tomoto::non_topic_id) return -1;
			return doc->path[x]; 
		});
	});
}


PyObject* HLDA_getAlpha(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IHLDAModel*>(self->inst);
		vector<float> ret;
		for (size_t i = 0; i < inst->getLevelDepth(); ++i)
		{
			ret.emplace_back(inst->getAlpha(i));
		}
		return py::buildPyValue(ret);
	});
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
	{ "loads", (PyCFunction)HLDA_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "is_live_topic", (PyCFunction)HLDA_isLiveTopic, METH_VARARGS | METH_KEYWORDS, HLDA_is_live_topic__doc__ },
	{ "num_docs_of_topic", (PyCFunction)HLDA_getNumDocsOfTopic, METH_VARARGS | METH_KEYWORDS, HLDA_num_docs_of_topic__doc__ },
	{ "level", (PyCFunction)HLDA_getLevelOfTopic, METH_VARARGS | METH_KEYWORDS, HLDA_level__doc__ },
	{ "parent_topic", (PyCFunction)HLDA_getParentTopicId, METH_VARARGS | METH_KEYWORDS, HLDA_parent_topic__doc__ },
	{ "children_topics", (PyCFunction)HLDA_getChildTopicId, METH_VARARGS | METH_KEYWORDS, HLDA_children_topics__doc__ },
	{ nullptr }
};

static PyGetSetDef HLDA_getseters[] = {
	{ (char*)"alpha", (getter)HLDA_getAlpha, nullptr, LDA_alpha__doc__, nullptr },
	{ (char*)"gamma", (getter)HLDA_getGamma, nullptr, HLDA_gamma__doc__, nullptr },
	{ (char*)"live_k", (getter)HLDA_getLiveK, nullptr, HLDA_live_k__doc__, nullptr },
	{ (char*)"depth", (getter)HLDA_getLevelDepth, nullptr, HLDA_depth__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject HLDA_type = { {
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
} };

DEFINE_DOCUMENT_GETTER(tomoto::DocumentHLDA, path, path);
