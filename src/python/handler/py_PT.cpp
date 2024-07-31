#include "../TopicModel/PT.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int PT_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::PTArgs margs;

	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "p", "alpha", "eta", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnOfOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &margs.p, &objAlpha, &margs.eta, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		if (margs.p == 0) margs.p = margs.k * 10;

		tomoto::ITopicModel* inst = tomoto::IPTModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, margs.k, margs.p, margs.alpha, margs.eta, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

DEFINE_GETTER(tomoto::IPTModel, PT, getP);

DEFINE_LOADER(PT, PT_type);

static PyMethodDef PT_methods[] =
{
	{ "load", (PyCFunction)PT_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)PT_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ nullptr }
};


static PyGetSetDef PT_getseters[] = {
	{ (char*)"p", (getter)PT_getP, nullptr, PT_p__doc__, nullptr },
	{ nullptr },
};

DEFINE_DOCUMENT_GETTER(tomoto::DocumentPT, pseudo_doc_id, pseudoDoc);


TopicModelTypeObject PT_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.PTModel",             /* tp_name */
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
	PT___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	PT_methods,             /* tp_methods */
	0,						 /* tp_members */
	PT_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PT_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
}};


PyObject* Document_getTopicsFromPseudoDoc(DocumentObject* self, size_t topN)
{
	tomoto::IPTModel* mdl = dynamic_cast<tomoto::IPTModel*>(self->corpus->tm->inst);
	if (!mdl) throw py::ValueError{ "`from_pseudo_doc` is valid for only `tomotopy.PTModel`." };
	return py::buildPyValue(self->corpus->tm->inst->getTopicsByDocSorted(self->getBoundDoc(), topN));
}

PyObject* Document_getTopicDistFromPseudoDoc(DocumentObject* self, bool normalize)
{
	tomoto::IPTModel* mdl = dynamic_cast<tomoto::IPTModel*>(self->corpus->tm->inst);
	if (!mdl) throw py::ValueError{ "`from_pseudo_doc` is valid for only `tomotopy.PTModel`." };
	return py::buildPyValue(self->corpus->tm->inst->getTopicsByDoc(self->getBoundDoc(), !!normalize));
}