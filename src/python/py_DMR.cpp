#include "../TopicModel/DMR.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType DMR_misc_args(const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["metadata"] = getValueFromMiscDefault<std::string>("metadata", o, "`DMRModel` needs a `metadata` value in `str` type.");
	return ret;
}

static int DMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "alpha", "eta", "sigma", "alpha_epsilon", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnffffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&K, &alpha, &eta, &sigma, &alphaEpsilon, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IDMRModel::create((tomoto::TermWeight)tw, K, alpha, sigma, eta, alphaEpsilon, seed);
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, K, alpha, eta, sigma, alphaEpsilon, seed
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

static PyObject* DMR_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] 'words' should be an iterable of str.");
		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["metadata"] = metadata;
		auto ret = inst->addDoc(raw);
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

static DocumentObject* DMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] 'words' should be an iterable of str.");
		tomoto::RawDoc raw = buildRawDoc(argWords);
		raw.misc["metadata"] = metadata;
		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
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

static VocabObject* DMR_getMetadataDict(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&static_cast<tomoto::IDMRModel*>(self->inst)->getMetadataDict();
		ret->size = -1;
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

static PyObject* DMR_getLambda(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getF() };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getLambdaByTopic(i);
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

static PyObject* DMR_getAlpha(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getF() };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getLambdaByTopic(i);
			Eigen::Map<Eigen::ArrayXf> ml{ l.data(), (Eigen::Index)l.size() };
			ml = ml.exp();
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

DEFINE_GETTER(tomoto::IDMRModel, DMR, getAlphaEps);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getSigma);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getF);

PyObject* Document_DMR_metadata(DocumentObject * self, void* closure)
{
	try
	{
		if (self->corpus->isIndependent()) return nullptr;
		if (!self->doc) throw runtime_error{ "doc is null!" };
		auto inst = (tomoto::IDMRModel*)self->corpus->tm->inst;
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDMR<tomoto::TermWeight::one>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(inst->getMetadataDict().toWord(doc->metadata));
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDMR<tomoto::TermWeight::idf>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(inst->getMetadataDict().toWord(doc->metadata));
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentDMR<tomoto::TermWeight::pmi>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(inst->getMetadataDict().toWord(doc->metadata));
		} while (0);
		return nullptr;
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

DEFINE_LOADER(DMR, DMR_type);

static PyMethodDef DMR_methods[] =
{
	{ "add_doc", (PyCFunction)DMR_addDoc, METH_VARARGS | METH_KEYWORDS, DMR_add_doc__doc__ },
	{ "make_doc", (PyCFunction)DMR_makeDoc, METH_VARARGS | METH_KEYWORDS, DMR_make_doc__doc__ },
	{ "load", (PyCFunction)DMR_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ nullptr }
};

static PyGetSetDef DMR_getseters[] = {
	{ (char*)"f", (getter)DMR_getF, nullptr, DMR_f__doc__, nullptr },
	{ (char*)"sigma", (getter)DMR_getSigma, nullptr, DMR_sigma__doc__, nullptr },
	{ (char*)"alpha_epsilon", (getter)DMR_getAlphaEps, nullptr, DMR_alpha_epsilon__doc__, nullptr },
	{ (char*)"metadata_dict", (getter)DMR_getMetadataDict, nullptr, DMR_metadata_dict__doc__, nullptr },
	{ (char*)"lambdas", (getter)DMR_getLambda, nullptr, DMR_lamdas__doc__, nullptr },
	{ (char*)"alpha", (getter)DMR_getAlpha, nullptr, DMR_alpha__doc__, nullptr },
	{ nullptr },
};


TopicModelTypeObject DMR_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.DMRModel",             /* tp_name */
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
	DMR___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	DMR_methods,             /* tp_methods */
	0,						 /* tp_members */
	DMR_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)DMR_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
}, DMR_misc_args};