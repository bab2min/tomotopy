#include "../TopicModel/SLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType SLDA_misc_args(const tomoto::RawDoc::MiscType& o)
{
	tomoto::RawDoc::MiscType ret;
	ret["y"] = getValueFromMiscDefault<vector<float>>("y", o, "`SLDAModel` needs a `y` value in `Iterable[float]` type.");
	return ret;
}

static int SLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1f, eta = 0.01f;
	PyObject *vars = nullptr, *mu = nullptr, *nuSq = nullptr, *glmCoef = nullptr;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k",
		"vars", "alpha", "eta",
		"mu", "nu_sq", "glm_param", "seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnOffOOOnOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &K, 
		&vars, &alpha, &eta, 
		&mu, &nuSq, &glmCoef, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		vector<tomoto::ISLDAModel::GLM> varTypes;
		vector<string> varTypeStrs;
		if (vars)
		{
			varTypeStrs = py::toCpp<vector<string>>(vars, "`vars` must be an iterable.");
			for (auto& s : varTypeStrs)
			{
				tomoto::ISLDAModel::GLM t;
				if (s == "l") t = tomoto::ISLDAModel::GLM::linear;
				else if (s == "b") t = tomoto::ISLDAModel::GLM::binary_logistic;
				else throw runtime_error{ "Unknown var type '" + s + "'" };
				varTypes.emplace_back(t);
			}
		}
		
		vector<tomoto::Float> vmu, vnuSq, vglmCoef; 
		float fTemp;
		if (mu)
		{
			if ((fTemp = (float)PyFloat_AsDouble(mu)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				vmu = py::toCpp<vector<tomoto::Float>>(mu, "`mu` must be float or iterable of float.");
			}
			else
			{
				vmu.resize(varTypes.size(), fTemp);
			}
		}

		if (nuSq)
		{
			if ((fTemp = (float)PyFloat_AsDouble(nuSq)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				vnuSq = py::toCpp<vector<tomoto::Float>>(nuSq, "`nu_sq` must be float or iterable of float.");
			}
			else
			{
				vnuSq.resize(varTypes.size(), fTemp);
			}
		}

		if (glmCoef)
		{
			if ((fTemp = (float)PyFloat_AsDouble(glmCoef)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				vglmCoef = py::toCpp<vector<tomoto::Float>>(glmCoef, "`glm_param` must be float or iterable of float.");
			}
			else
			{
				vglmCoef.resize(varTypes.size(), fTemp);
			}
		}

		tomoto::ITopicModel* inst = tomoto::ISLDAModel::create((tomoto::TermWeight)tw, K, varTypes, 
			alpha, eta, vmu, vnuSq, vglmCoef,
			seed);
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, K, varTypeStrs, alpha, eta,
			vmu, vnuSq, vglmCoef
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

static PyObject* SLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] 'words' should be an iterable of str.");
		tomoto::RawDoc raw = buildRawDoc(argWords);

		if (argY)
		{
			raw.misc["y"] = py::toCpp<vector<tomoto::Float>>(argY, "`y` must be an iterable of float.");
		}
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

static DocumentObject* SLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] 'words' should be an iterable of str.");
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

static PyObject* SLDA_getRegressionCoef(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argVarId = nullptr;
	static const char* kwlist[] = { "var_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", (char**)kwlist, &argVarId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
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
		if (varId >= inst->getF()) throw runtime_error{ "'var_id' must be < 'f'" };
		return py::buildPyValue(inst->getRegressionCoef(varId));
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

static PyObject* SLDA_getTypeOfVar(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t varId;
	static const char* kwlist[] = { "var_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &varId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (varId >= inst->getF()) throw runtime_error{ "'var_id' must be < 'f'" };
		return py::buildPyValue(std::string{ "l\0b" + (size_t)inst->getTypeOfVar(varId) * 2 });
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

static PyObject* SLDA_estimateVars(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* argDoc;
	static const char* kwlist[] = { "doc", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argDoc)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (py::UniqueObj iter = py::UniqueObj{ PyObject_GetIter(argDoc) })
		{
			py::UniqueObj nextDoc;
			std::vector<const tomoto::DocumentBase*> docs;
			while ((nextDoc = py::UniqueObj{ PyIter_Next(iter) }))
			{
				if (!PyObject_TypeCheck(nextDoc, &UtilsDocument_type)) throw runtime_error{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
				auto* doc = (DocumentObject*)nextDoc.get();
				if (doc->corpus->tm != self) throw runtime_error{ "`doc` was from another model, not fit to this model" };
				docs.emplace_back(doc->getBoundDoc());
			}
			if (PyErr_Occurred()) return nullptr;
			return py::buildPyValueTransform(docs.begin(), docs.end(), [&](const tomoto::DocumentBase* d)
			{
				return inst->estimateVars(d);
			});
		}
		else
		{
			PyErr_Clear();
		}

		if (!PyObject_TypeCheck(argDoc, &UtilsDocument_type)) throw runtime_error{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
		auto* doc = (DocumentObject*)argDoc;
		if (doc->corpus->tm != self) throw runtime_error{ "`doc` was from another model, not fit to this model" };

		return py::buildPyValue(inst->estimateVars(doc->getBoundDoc()));
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


DEFINE_GETTER(tomoto::ISLDAModel, SLDA, getF);

DEFINE_DOCUMENT_GETTER(tomoto::DocumentSLDA, y, y);

DEFINE_LOADER(SLDA, SLDA_type);


static PyMethodDef SLDA_methods[] =
{
	{ "load", (PyCFunction)SLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
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
