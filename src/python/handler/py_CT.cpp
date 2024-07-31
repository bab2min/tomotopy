#include "../TopicModel/CT.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int CT_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::CTArgs margs;

	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "smoothing_alpha", "eta", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOfOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &objAlpha, &margs.eta, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`smoothing_alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::ICTModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, margs.k, margs.alpha, margs.eta, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* CT_getCorrelations(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argTopicId = nullptr;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist, &argTopicId)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::ICTModel*>(self->inst);

		if (!argTopicId || argTopicId == Py_None)
		{
			npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getK() };
			PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
			for (size_t i = 0; i < inst->getK(); ++i)
			{
				auto l = inst->getCorrelationTopic(i);
				memcpy(PyArray_GETPTR2((PyArrayObject*)ret, i, 0), l.data(), sizeof(float) * l.size());
			}
			return ret;
		}

		size_t topicId = PyLong_AsLong(argTopicId);
		if (topicId == (size_t)-1 && PyErr_Occurred()) return nullptr;
		if (topicId >= inst->getK()) throw py::ValueError{ "`topic_id` must be in range [0, `k`)" };
		return py::buildPyValue(inst->getCorrelationTopic(topicId));
	});
}

DEFINE_GETTER(tomoto::ICTModel, CT, getNumBetaSample);
DEFINE_GETTER(tomoto::ICTModel, CT, getNumTMNSample);
DEFINE_GETTER(tomoto::ICTModel, CT, getPriorMean);

PyObject* CT_getPriorCov(TopicModelObject *self, void *closure)
{
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::ICTModel*>(self->inst);
		py::UniqueObj obj{ py::buildPyValue(inst->getPriorCov()) };
		PyArray_Dims dims;
		npy_intp d[2] = { (npy_intp)self->inst->getK(), (npy_intp)self->inst->getK() };
		dims.ptr = d;
		dims.len = 2;
		return PyArray_Newshape((PyArrayObject*)obj.get(), &dims, NPY_CORDER);
	});
}

DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ICTModel, CT, setNumBetaSample);
DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ICTModel, CT, setNumTMNSample);

DEFINE_LOADER(CT, CT_type);


static PyMethodDef CT_methods[] =
{
	{ "load", (PyCFunction)CT_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)CT_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "get_correlations", (PyCFunction)CT_getCorrelations, METH_VARARGS | METH_KEYWORDS, CT_get_correlations__doc__ },
	{ nullptr }
};


static PyGetSetDef CT_getseters[] = {
	{ (char*)"num_beta_sample", (getter)CT_getNumBetaSample, (setter)CT_setNumBetaSample, CT_num_beta_sample__doc__, nullptr },
	{ (char*)"num_tmn_sample", (getter)CT_getNumTMNSample, (setter)CT_setNumTMNSample, CT_num_tmn_sample__doc__, nullptr },
	{ (char*)"prior_mean", (getter)CT_getPriorMean, nullptr, CT_prior_mean__doc__, nullptr },
	{ (char*)"prior_cov", (getter)CT_getPriorCov, nullptr, CT_prior_cov__doc__, nullptr },
	{ (char*)"alpha", nullptr, nullptr, CT_alpha__doc__, nullptr },
	{ nullptr },
};


TopicModelTypeObject CT_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.CTModel",             /* tp_name */
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
	CT___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	CT_methods,             /* tp_methods */
	0,						 /* tp_members */
	CT_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)CT_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
}};


PyObject* Document_beta(DocumentObject* self, void* closure)
{
	return py::handleExc([&]() -> PyObject*
	{
		if (self->corpus->isIndependent()) throw py::AttributeError{ "doc has no `beta` field!" };
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };

		if (auto ret = docVisit<tomoto::DocumentCTM>(self->getBoundDoc(), [](auto* doc)
		{
			return py::buildPyValueTransform(
				doc->smBeta.data(), doc->smBeta.data() + doc->smBeta.size(),
				logf
			);
		})) return ret;
		throw py::AttributeError{ "doc has no `beta` field!" };
	});
}
