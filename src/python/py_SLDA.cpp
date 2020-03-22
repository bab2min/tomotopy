#include "../TopicModel/SLDA.h"

#include "module.h"

using namespace std;

static int SLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01;
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
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };

		}
		vector<tomoto::ISLDAModel::GLM> varTypes;
		if (vars)
		{
			py::UniqueObj iter;
			if (!(iter = PyObject_GetIter(vars))) throw runtime_error{ "'vars' must be an iterable." };
			auto vs = py::makeIterToVector<string>(iter);
			for (auto& s : vs)
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
			if ((fTemp = PyFloat_AsDouble(mu)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				py::UniqueObj iter;
				if (!(iter = PyObject_GetIter(mu))) throw runtime_error{ "'mu' must be float or iterable of float." };

				vmu = py::makeIterToVector<tomoto::Float>(iter);
			}
			else
			{
				vmu.resize(varTypes.size(), fTemp);
			}
		}

		if (nuSq)
		{
			if ((fTemp = PyFloat_AsDouble(nuSq)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				py::UniqueObj iter;
				if (!(iter = PyObject_GetIter(nuSq))) throw runtime_error{ "'nu_sq' must be float or iterable of float." };

				vnuSq = py::makeIterToVector<tomoto::Float>(iter);
			}
			else
			{
				vnuSq.resize(varTypes.size(), fTemp);
			}
		}

		if (glmCoef)
		{
			if ((fTemp = PyFloat_AsDouble(glmCoef)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				py::UniqueObj iter;
				if (!(iter = PyObject_GetIter(glmCoef))) throw runtime_error{ "'glm_param' must be float or iterable of float." };

				vglmCoef = py::makeIterToVector<tomoto::Float>(iter);
			}
			else
			{
				vglmCoef.resize(varTypes.size(), fTemp);
			}
		}

		tomoto::ITopicModel* inst = tomoto::ISLDAModel::create((tomoto::TermWeight)tw, K, varTypes, 
			alpha, eta, vmu, vnuSq, vglmCoef,
			tomoto::RandGen{ seed });
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
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "'words' must be an iterable of str." };
		}
		auto words = py::makeIterToVector<string>(iter);
		vector<tomoto::Float> ys;
		if (argY)
		{
			py::UniqueObj iter2;
			if (!(iter2 = PyObject_GetIter(argY))) throw runtime_error{ "'y' must be an iterable of float." };
			ys = py::makeIterToVector<tomoto::Float>(iter2);
		}
		auto ret = inst->addDoc(words, ys);
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

static PyObject* SLDA_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr, *argY = nullptr;
	const char* argRaw = nullptr;
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOOO", (char**)kwlist,
		&argWords, &argRaw, &argStartPos, &argLength, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;

		py::UniqueObj iter = PyObject_GetIter(argWords);
		vector<tomoto::Vid> words = py::makeIterToVector<tomoto::Vid>(iter);
		iter = PyObject_GetIter(argStartPos);
		vector<uint32_t> startPos = py::makeIterToVector<uint32_t>(iter);
		iter = PyObject_GetIter(argLength);
		vector<uint16_t> length = py::makeIterToVector<uint16_t>(iter);
		char2Byte(raw, startPos, length);
		vector<tomoto::Float> ys;
		if (argY)
		{
			py::UniqueObj iter2;
			if (!(iter2 = PyObject_GetIter(argY))) throw runtime_error{ "'y' must be an iterable of float." };
			ys = py::makeIterToVector<tomoto::Float>(iter2);
		}

		auto ret = inst->addDoc(raw, words, startPos, length, ys);
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

static PyObject* SLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto words = py::makeIterToVector<string>(iter);
		vector<tomoto::Float> ys;
		if (argY)
		{
			py::UniqueObj iter2;
			if (!(iter2 = PyObject_GetIter(argY))) throw runtime_error{ "'y' must be an iterable of float." };
			ys = py::makeIterToVector<tomoto::Float>(iter2);
		}
		auto ret = inst->makeDoc(words, ys);
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

static PyObject* SLDA_getRegressionCoef(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t varId;
	static const char* kwlist[] = { "var_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &varId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
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
		return py::buildPyValue("l\0b" + (size_t)inst->getTypeOfVar(varId) * 2);
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
		if (py::UniqueObj iter = PyObject_GetIter(argDoc))
		{
			py::UniqueObj nextDoc;
			std::vector<const tomoto::DocumentBase*> docs;
			while ((nextDoc = PyIter_Next(iter)))
			{
				if (Py_TYPE(nextDoc) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document or list of tomotopy.Document" };
				docs.emplace_back(((DocumentObject*)nextDoc.get())->doc);
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

		if (Py_TYPE(argDoc) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document or list of tomotopy.Document" };
		auto* doc = (DocumentObject*)argDoc;
		if (doc->parentModel != self) throw runtime_error{ "'doc' was from another model, not fit to this model" };

		return py::buildPyValue(inst->estimateVars(doc->doc));
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
	{ "_add_doc", (PyCFunction)SLDA_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
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


PyTypeObject SLDA_type = {
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
};
