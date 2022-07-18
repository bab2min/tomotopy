#include "../TopicModel/DMR.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType DMR_misc_args(TopicModelObject* self, const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["metadata"] = getValueFromMiscDefault<string>("metadata", o, "`DMRModel` needs a `metadata` value in `str` type.");
	ret["multi_metadata"] = getValueFromMiscDefault<vector<string>>("multi_metadata", o, "`DMRModel` needs a `multi_metadata` value in `List[str]` type.");
	return ret;
}

static int DMR_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::DMRArgs margs;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "alpha", "eta", "sigma", "alpha_epsilon", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOfffOOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&margs.k, &objAlpha, &margs.eta, &margs.sigma, &margs.alphaEps, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		tomoto::ITopicModel* inst = tomoto::IDMRModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, margs.k, margs.alpha, margs.eta, margs.sigma, margs.alphaEps, margs.seed
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* DMR_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argWords;
	PyObject* multiMetadata = nullptr;
	const char* metadata = nullptr;
	size_t ignoreEmptyWords = 1;
	static const char* kwlist[] = { "words", "metadata", "multi_metadata", "ignore_empty_words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|zOp", (char**)kwlist, 
		&argWords, &metadata, &multiMetadata, &ignoreEmptyWords)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (self->isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		if (multiMetadata && PyUnicode_Check(multiMetadata))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);
		if (!metadata) metadata = "";
		raw.misc["metadata"] = metadata;
		if (multiMetadata)
		{
			raw.misc["multi_metadata"] = py::toCpp<vector<string>>(multiMetadata,
				[=]() { return "`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"; }
			);
		}
		try
		{
			auto ret = inst->addDoc(raw);
			return py::buildPyValue(ret);
		}
		catch (const tomoto::exc::EmptyWordArgument&)
		{
			if (ignoreEmptyWords)
			{
				Py_INCREF(Py_None);
				return Py_None;
			}
			else
			{
				throw;
			}
		}
	});
}

static DocumentObject* DMR_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argWords;
	PyObject* multiMetadata = nullptr;
	const char* metadata = nullptr;
	static const char* kwlist[] = { "words", "metadata", "multi_metadata", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|zO", (char**)kwlist, 
		&argWords, &metadata, &multiMetadata)) return nullptr;
	return py::handleExc([&]() -> DocumentObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (!self->isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		if (multiMetadata && PyUnicode_Check(multiMetadata))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);
		if (!metadata) metadata = "";
		raw.misc["metadata"] = metadata;
		if (multiMetadata)
		{
			raw.misc["multi_metadata"] = py::toCpp<vector<string>>(multiMetadata,
				[=]() { return "`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"; }
			);
		}
		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
		return ret;
	});
}

static PyObject* DMR_getTopicPrior(TopicModelObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject* multiMetadata = nullptr;
	const char* metadata = nullptr;
	size_t raw = 0;
	static const char* kwlist[] = { "metadata", "multi_metadata", "raw", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|zOp", (char**)kwlist,
		&metadata, &multiMetadata, &raw)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		if (multiMetadata && PyUnicode_Check(multiMetadata))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) return nullptr;
		}
		if (!metadata) metadata = "";

		vector<string> multiMd;
		if (multiMetadata)
		{
			multiMd = py::toCpp<vector<string>>(multiMetadata,
				[=]() { return "`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"; }
			);
		}
		return py::buildPyValue(inst->getTopicPrior(metadata, multiMd, !!raw));
	});
}

static VocabObject* DMR_getMetadataDict(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&static_cast<tomoto::IDMRModel*>(self->inst)->getMetadataDict();
		ret->size = -1;
		return ret;
	});
}

static VocabObject* DMR_getMultiMetadataDict(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&static_cast<tomoto::IDMRModel*>(self->inst)->getMultiMetadataDict();
		ret->size = -1;
		return ret;
	});
}

static PyObject* DMR_getLambda(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)(inst->getF() * inst->getMdVecSize()) };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getLambdaByTopic(i);
			memcpy(PyArray_GETPTR2((PyArrayObject*)ret, i, 0), l.data(), sizeof(float) * shapes[1]);
		}
		return ret;
	});
}

static PyObject* DMR_getLambdaV2(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		npy_intp shapes[3] = { (npy_intp)inst->getK(), (npy_intp)inst->getF(), (npy_intp)inst->getMdVecSize() };
		PyObject* ret = PyArray_EMPTY(3, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getLambdaByTopic(i);
			memcpy(PyArray_GETPTR3((PyArrayObject*)ret, i, 0, 0), l.data(), sizeof(float) * shapes[1] * shapes[2]);
		}
		return ret;
	});
}

static PyObject* DMR_getAlpha(TopicModelObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::IDMRModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getF() };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getLambdaByTopic(i);
			Eigen::Map<Eigen::ArrayXf> ml{ l.data(), (Eigen::Index)l.size() };
			ml = ml.exp() + inst->getAlphaEps();
			memcpy(PyArray_GETPTR2((PyArrayObject*)ret, i, 0), l.data(), sizeof(float) * shapes[1]);
		}
		return ret;
	});
}

DEFINE_GETTER(tomoto::IDMRModel, DMR, getAlphaEps);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getSigma);
DEFINE_GETTER(tomoto::IDMRModel, DMR, getF);

PyObject* Document_DMR_metadata(DocumentObject * self, void* closure)
{
	return py::handleExc([&]() -> PyObject*
	{
		if (self->corpus->isIndependent()) return nullptr;
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };
		auto inst = (tomoto::IDMRModel*)self->corpus->tm->inst;

		return docVisit<tomoto::DocumentDMR>(self->getBoundDoc(), [&](auto* doc)
		{
			return py::buildPyValue(inst->getMetadataDict().toWord(doc->metadata));
		});
	});
}

PyObject* Document_DMR_multiMetadata(DocumentObject* self, void* closure)
{
	return py::handleExc([&]()
	{
		if (!self->doc) throw py::RuntimeError{ "doc is null!" };
		auto inst = (tomoto::IDMRModel*)self->corpus->tm->inst;
		
		if(auto* ret = docVisit<tomoto::DocumentDMR>(self->getBoundDoc(), [&](auto* doc)
		{
			return py::buildPyValueTransform(doc->multiMetadata.begin(), doc->multiMetadata.end(), [&](uint64_t x)
			{
				return inst->getMultiMetadataDict().toWord(x);
			});
		})) return ret;
		throw py::AttributeError{ "doc has no `multi_metadata` field!" };
	});
}

DEFINE_LOADER(DMR, DMR_type);

static PyMethodDef DMR_methods[] =
{
	{ "add_doc", (PyCFunction)DMR_addDoc, METH_VARARGS | METH_KEYWORDS, DMR_add_doc__doc__ },
	{ "make_doc", (PyCFunction)DMR_makeDoc, METH_VARARGS | METH_KEYWORDS, DMR_make_doc__doc__ },
	{ "load", (PyCFunction)DMR_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)DMR_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "get_topic_prior", (PyCFunction)DMR_getTopicPrior,  METH_VARARGS | METH_KEYWORDS, DMR_get_topic_prior__doc__ },
	{ nullptr }
};

static PyGetSetDef DMR_getseters[] = {
	{ (char*)"f", (getter)DMR_getF, nullptr, DMR_f__doc__, nullptr },
	{ (char*)"sigma", (getter)DMR_getSigma, nullptr, DMR_sigma__doc__, nullptr },
	{ (char*)"alpha_epsilon", (getter)DMR_getAlphaEps, nullptr, DMR_alpha_epsilon__doc__, nullptr },
	{ (char*)"metadata_dict", (getter)DMR_getMetadataDict, nullptr, DMR_metadata_dict__doc__, nullptr },
	{ (char*)"multi_metadata_dict", (getter)DMR_getMultiMetadataDict, nullptr, DMR_multi_metadata_dict__doc__, nullptr },
	{ (char*)"lambdas", (getter)DMR_getLambda, nullptr, DMR_lamdas__doc__, nullptr },
	{ (char*)"lambda_", (getter)DMR_getLambdaV2, nullptr, DMR_lamda___doc__, nullptr },
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