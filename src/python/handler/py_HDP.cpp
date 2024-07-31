#include "../TopicModel/HDP.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int HDP_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::HDPArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "initial_k", "alpha", "eta", "gamma", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnfffOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &margs.alpha[0], &margs.eta, &margs.gamma, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::IHDPModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, margs.k, margs.alpha[0], margs.eta, margs.gamma, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* HDP_isLiveTopic(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IHDPModel*>(self->inst);
		if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < K" };

		return py::buildPyValue(inst->isLiveTopic(topicId));
	});
}

static PyObject* HDP_convertToLDA(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	float topicThreshold = 0;
	static const char* kwlist[] = { "topic_threshold", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|f", (char**)kwlist, &topicThreshold)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto inst = static_cast<tomoto::IHDPModel*>(self->inst);
		std::vector<tomoto::Tid> newK;
		auto lda = inst->convertToLDA(topicThreshold, newK);
		py::UniqueObj r{ PyObject_CallObject((PyObject*)&LDA_type, nullptr) };
		auto ret = (TopicModelObject*)r.get();
		delete ret->inst;
		ret->inst = lda.release();
		ret->isPrepared = true;
		ret->minWordCnt = self->minWordCnt;
		ret->minWordDf = self->minWordDf;
		ret->removeTopWord = self->removeTopWord;
		return Py_BuildValue("(NN)", r.release(), py::buildPyValue(newK, py::cast_to_signed));
	});
}

static PyObject* HDP_purgeDeadTopics(TopicModelObject* self, PyObject*)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto inst = static_cast<tomoto::IHDPModel*>(self->inst);
		std::vector<int32_t> ret;
		for (auto t : inst->purgeDeadTopics())
		{
			ret.emplace_back((int16_t)t);
		}
		return py::buildPyValue(ret);
	});
}

PyObject* Document_HDP_Z(DocumentObject* self, void* closure)
{
	return docVisit<tomoto::DocumentHDP>(self->getBoundDoc(), [](auto* doc)
	{
		return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](tomoto::Tid x) -> int16_t
		{ 
			if (x == tomoto::non_topic_id) return -1;
			return doc->numTopicByTable[x].topic; 
		});
	});
}


DEFINE_GETTER(tomoto::IHDPModel, HDP, getAlpha);
DEFINE_GETTER(tomoto::IHDPModel, HDP, getGamma);
DEFINE_GETTER(tomoto::IHDPModel, HDP, getTotalTables);
DEFINE_GETTER(tomoto::IHDPModel, HDP, getLiveK);

DEFINE_LOADER(HDP, HDP_type);

static PyMethodDef HDP_methods[] =
{
	{ "load", (PyCFunction)HDP_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)HDP_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "is_live_topic", (PyCFunction)HDP_isLiveTopic, METH_VARARGS | METH_KEYWORDS, HDP_is_live_topic__doc__ },
	{ "convert_to_lda", (PyCFunction)HDP_convertToLDA, METH_VARARGS | METH_KEYWORDS, HDP_convert_to_lda__doc__ },
	{ "purge_dead_topics", (PyCFunction)HDP_purgeDeadTopics, METH_NOARGS, HDP_purge_dead_topics__doc__ },
	{ nullptr }
};

static PyGetSetDef HDP_getseters[] = {
	{ (char*)"alpha", (getter)HDP_getAlpha, nullptr, LDA_alpha__doc__, nullptr },
	{ (char*)"gamma", (getter)HDP_getGamma, nullptr, HDP_gamma__doc__, nullptr },
	{ (char*)"live_k", (getter)HDP_getLiveK, nullptr, HDP_live_k__doc__, nullptr },
	{ (char*)"num_tables", (getter)HDP_getTotalTables, nullptr, HDP_num_tables__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject HDP_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.HDPModel",             /* tp_name */
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
	HDP___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	HDP_methods,             /* tp_methods */
	0,						 /* tp_members */
	HDP_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)HDP_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
} };