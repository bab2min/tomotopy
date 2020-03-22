#include "../TopicModel/DMR.h"

#include "module.h"

using namespace std;

static int DMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "alpha", "eta", "sigma", "alpha_epsilon", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnffffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&K, &alpha, &eta, &sigma, &alphaEpsilon, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		tomoto::ITopicModel* inst = tomoto::IDMRModel::create((tomoto::TermWeight)tw, K, alpha, sigma, eta, alphaEpsilon, tomoto::RandGen{ seed });
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
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto ret = inst->addDoc(py::makeIterToVector<string>(iter), { string{metadata} });
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

static PyObject* DMR_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr;
	const char* argRaw = nullptr;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOOs", (char**)kwlist,
		&argWords, &argRaw, &argStartPos, &argLength, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;

		py::UniqueObj iter = PyObject_GetIter(argWords);
		vector<tomoto::Vid> words = py::makeIterToVector<tomoto::Vid>(iter);
		iter = PyObject_GetIter(argStartPos);
		vector<uint32_t> startPos = py::makeIterToVector<uint32_t>(iter);
		iter = PyObject_GetIter(argLength);
		vector<uint16_t> length = py::makeIterToVector<uint16_t>(iter);
		char2Byte(raw, startPos, length);
		auto ret = inst->addDoc(raw, words, startPos, length, { string{metadata} });
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

static PyObject* DMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	const char* metadata = "";
	static const char* kwlist[] = { "words", "metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|s", (char**)kwlist, &argWords, &metadata)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter), { string{metadata} });
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

static PyObject* DMR_getMetadataDict(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		py::UniqueObj args = Py_BuildValue("(On)", self,
			&static_cast<tomoto::IDMRModel*>(self->inst)->getMetadataDict());
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

static PyObject* DMR_getLambda(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		PyObject* ret = PyList_New(inst->getK());
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			PyList_SetItem(ret, i, py::buildPyValue(inst->getLambdaByTopic(i)));
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

DEFINE_DOCUMENT_GETTER(tomoto::DocumentDMR, metadata, metadata);

DEFINE_LOADER(DMR, DMR_type);

static PyMethodDef DMR_methods[] =
{
	{ "add_doc", (PyCFunction)DMR_addDoc, METH_VARARGS | METH_KEYWORDS, DMR_add_doc__doc__ },
	{ "_add_doc", (PyCFunction)DMR_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
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
	{ nullptr },
};


PyTypeObject DMR_type = {
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
};