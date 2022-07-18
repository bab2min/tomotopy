#include "../TopicModel/SLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType SLDA_misc_args(TopicModelObject* self, const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["y"] = getValueFromMiscDefault<vector<float>>("y", o, "`SLDAModel` requires a `y` value in `Iterable[float]` type.");
	return ret;
}

static int SLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	tomoto::SLDAArgs margs;
	PyObject *vars = nullptr, *mu = nullptr, *nuSq = nullptr, *glmCoef = nullptr;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	PyObject* objAlpha = nullptr, *objSeed = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k",
		"vars", "alpha", "eta",
		"mu", "nu_sq", "glm_param", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOOfOOOOOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &margs.k,
		&vars, &objAlpha, &margs.eta,
		&mu, &nuSq, &glmCoef, &objSeed, &objCorpus, &objTransform)) return -1;
	return py::handleExc([&]()
	{
		if (objAlpha) margs.alpha = broadcastObj<tomoto::Float>(objAlpha, margs.k,
			[=]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(objAlpha) + ")"; }
		);
		if (objSeed) margs.seed = py::toCpp<size_t>(objSeed, "`seed` must be an integer or None.");

		vector<string> varTypeStrs;
		if (vars)
		{
			varTypeStrs = py::toCpp<vector<string>>(vars, "`vars` must be an iterable.");
			for (auto& s : varTypeStrs)
			{
				tomoto::ISLDAModel::GLM t;
				if (s == "l") t = tomoto::ISLDAModel::GLM::linear;
				else if (s == "b") t = tomoto::ISLDAModel::GLM::binary_logistic;
				else throw py::ValueError{ "Unknown var type '" + s + "'" };
				margs.vars.emplace_back(t);
			}
		}

		float fTemp;
		if (mu)
		{
			if ((fTemp = (float)PyFloat_AsDouble(mu)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				margs.mu = py::toCpp<vector<tomoto::Float>>(mu, "`mu` must be float or iterable of float.");
			}
			else
			{
				margs.mu.resize(varTypeStrs.size(), fTemp);
			}
		}

		if (nuSq)
		{
			if ((fTemp = (float)PyFloat_AsDouble(nuSq)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				margs.nuSq = py::toCpp<vector<tomoto::Float>>(nuSq, "`nu_sq` must be float or iterable of float.");
			}
			else
			{
				margs.nuSq.resize(varTypeStrs.size(), fTemp);
			}
		}

		if (glmCoef)
		{
			if ((fTemp = (float)PyFloat_AsDouble(glmCoef)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				margs.glmParam = py::toCpp<vector<tomoto::Float>>(glmCoef, "`glm_param` must be float or iterable of float.");
			}
			else
			{
				margs.glmParam.resize(varTypeStrs.size(), fTemp);
			}
		}

		tomoto::ITopicModel* inst = tomoto::ISLDAModel::create((tomoto::TermWeight)tw, margs);
		if (!inst) throw py::ValueError{ "unknown `tw` value" };
		self->inst = inst;
		self->isPrepared = false;
		self->seedGiven = !!objSeed;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop,
			margs.k, varTypeStrs, margs.alpha, margs.eta,
			margs.mu, margs.nuSq, margs.glmParam
		);
		py::setPyDictItem(self->initParams, "version", getVersion());

		insertCorpus(self, objCorpus, objTransform);
		return 0;
	});
}

static PyObject* SLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argY = nullptr;
	size_t ignoreEmptyWords = 1;
	static const char* kwlist[] = { "words", "y", "ignore_empty_words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|Op", (char**)kwlist, &argWords, &argY, &ignoreEmptyWords)) return nullptr;
	return py::handleExc([&]() -> PyObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (self->isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);

		if (argY)
		{
			raw.misc["y"] = py::toCpp<vector<tomoto::Float>>(argY, "`y` must be an iterable of float.");
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

static DocumentObject* SLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	return py::handleExc([&]() -> DocumentObject*
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		if (!self->isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) return nullptr;
		}
		tomoto::RawDoc raw = buildRawDoc(argWords);

		if (argY)
		{
			raw.misc["y"] = py::toCpp<vector<tomoto::Float>>(argY, "`y` must be an iterable of float.");
		}
		auto doc = inst->makeDoc(raw);
		py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		auto* ret = (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)&UtilsDocument_type, corpus.get(), nullptr);
		ret->doc = doc.release();
		ret->owner = true;
		return ret;
	});
}

static PyObject* SLDA_getRegressionCoef(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argVarId = nullptr;
	static const char* kwlist[] = { "var_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist, &argVarId)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (!argVarId || argVarId == Py_None)
		{
			npy_intp shapes[2] = { (npy_intp)inst->getF(), (npy_intp)inst->getK() };
			PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
			for (size_t i = 0; i < inst->getF(); ++i)
			{
				auto l = inst->getRegressionCoef(i);
				memcpy(PyArray_GETPTR2((PyArrayObject*)ret, i, 0), l.data(), sizeof(float) * l.size());
			}
			return ret;
		}

		size_t varId = PyLong_AsLong(argVarId);
		if (varId >= inst->getF()) throw py::ValueError{ "`var_id` must be < `f`" };
		return py::buildPyValue(inst->getRegressionCoef(varId));
	});
}

static PyObject* SLDA_getTypeOfVar(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t varId;
	static const char* kwlist[] = { "var_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &varId)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (varId >= inst->getF()) throw py::ValueError{ "`var_id` must be < `f`" };
		return py::buildPyValue(std::string{ "l\0b" + (size_t)inst->getTypeOfVar(varId) * 2 });
	});
}

static PyObject* SLDA_estimateVars(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argDoc;
	static const char* kwlist[] = { "doc", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argDoc)) return nullptr;
	return py::handleExc([&]()
	{
		if (!self->inst) throw py::RuntimeError{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		try
		{
			if (!PyObject_TypeCheck(argDoc, &UtilsDocument_type)) throw py::ConversionFail{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)argDoc;
			if (doc->corpus->tm != self) throw py::ConversionFail{ "`doc` was from another model, not fit to this model" };

			return py::buildPyValue(inst->estimateVars(doc->getBoundDoc()));
		}
		catch (const py::ConversionFail&)
		{
			PyErr_Clear();
		}

		py::UniqueObj iter = py::UniqueObj{ PyObject_GetIter(argDoc) };
		py::UniqueObj nextDoc;
		std::vector<const tomoto::DocumentBase*> docs;
		while ((nextDoc = py::UniqueObj{ PyIter_Next(iter) }))
		{
			if (!PyObject_TypeCheck(nextDoc, &UtilsDocument_type)) throw py::ConversionFail{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)nextDoc.get();
			if (doc->corpus->tm != self) throw py::ConversionFail{ "`doc` was from another model, not fit to this model" };
			docs.emplace_back(doc->getBoundDoc());
		}
		if (PyErr_Occurred()) throw py::ExcPropagation{};
		return py::buildPyValueTransform(docs.begin(), docs.end(), [&](const tomoto::DocumentBase* d)
		{
			return inst->estimateVars(d);
		});
	});
}


DEFINE_GETTER(tomoto::ISLDAModel, SLDA, getF);

DEFINE_DOCUMENT_GETTER(tomoto::DocumentSLDA, y, y);

DEFINE_LOADER(SLDA, SLDA_type);


static PyMethodDef SLDA_methods[] =
{
	{ "load", (PyCFunction)SLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "loads", (PyCFunction)SLDA_loads, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_loads__doc__ },
	{ "add_doc", (PyCFunction)SLDA_addDoc, METH_VARARGS | METH_KEYWORDS, SLDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)SLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, SLDA_make_doc__doc__},
	{ "get_regression_coef", (PyCFunction)SLDA_getRegressionCoef, METH_VARARGS | METH_KEYWORDS, SLDA_get_regression_coef__doc__},
	{ "get_var_type", (PyCFunction)SLDA_getTypeOfVar, METH_VARARGS | METH_KEYWORDS, SLDA_get_var_type__doc__},
	{ "estimate", (PyCFunction)SLDA_estimateVars, METH_VARARGS | METH_KEYWORDS, SLDA_estimate__doc__},
	{ nullptr }
};

static PyGetSetDef SLDA_getseters[] = {
	{ (char*)"f", (getter)SLDA_getF, nullptr, SLDA_f__doc__, nullptr },
	{ nullptr },
};


TopicModelTypeObject SLDA_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.SLDAModel",             /* tp_name */
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
	SLDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	SLDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	SLDA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)SLDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
}, SLDA_misc_args };
