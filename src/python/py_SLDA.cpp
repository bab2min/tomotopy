#include "../TopicModel/SLDA.h"

#include "module.h"

using namespace std;

static int SLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01;
	PyObject *vars = nullptr, *mu = nullptr, *nuSq = nullptr, *glmCoef = nullptr;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "rm_top", "k",
		"vars", "alpha", "eta",
		"mu", "nu_sq", "glm_param", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnOffOOOn", (char**)kwlist, 
		&tw, &minCnt, &rmTop, &K, 
		&vars, &alpha, &eta, 
		&mu, &nuSq, &glmCoef, &seed)) return -1;
	try
	{
		PyObject* iter;
		vector<tomoto::ISLDAModel::GLM> varTypes;
		if (vars)
		{
			if (!(iter = PyObject_GetIter(vars))) throw runtime_error{ "'vars' must be an iterable." };
			py::AutoReleaser ar{ iter };
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
		
		vector<tomoto::FLOAT> vmu, vnuSq, vglmCoef; 
		float fTemp;
		if (mu)
		{
			if ((fTemp = PyFloat_AsDouble(mu)) == -1 && PyErr_Occurred())
			{
				PyErr_Clear();
				if (!(iter = PyObject_GetIter(mu))) throw runtime_error{ "'mu' must be float or iterable of float." };

				py::AutoReleaser ar{ iter };
				vmu = py::makeIterToVector<tomoto::FLOAT>(iter);
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
				if (!(iter = PyObject_GetIter(nuSq))) throw runtime_error{ "'nu_sq' must be float or iterable of float." };

				py::AutoReleaser ar{ iter };
				vnuSq = py::makeIterToVector<tomoto::FLOAT>(iter);
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
				if (!(iter = PyObject_GetIter(glmCoef))) throw runtime_error{ "'glm_param' must be float or iterable of float." };

				py::AutoReleaser ar{ iter };
				vglmCoef = py::makeIterToVector<tomoto::FLOAT>(iter);
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
		self->removeTopWord = rmTop;
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
	PyObject *argWords, *iter = nullptr, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "'words' must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		auto words = py::makeIterToVector<string>(iter);
		vector<tomoto::FLOAT> ys;
		if (argY)
		{
			if (!(iter = PyObject_GetIter(argY))) throw runtime_error{ "'y' must be an iterable of float." };
			py::AutoReleaser arIter{ iter };
			ys = py::makeIterToVector<tomoto::FLOAT>(iter);
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

static PyObject* SLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr, *argY = nullptr;
	static const char* kwlist[] = { "words", "y", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argY)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ISLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		auto words = py::makeIterToVector<string>(iter);
		vector<tomoto::FLOAT> ys;
		if (argY)
		{
			if (!(iter = PyObject_GetIter(argY))) throw runtime_error{ "'y' must be an iterable of float." };
			py::AutoReleaser arIter{ iter };
			ys = py::makeIterToVector<tomoto::FLOAT>(iter);
		}
		auto ret = inst->makeDoc(words, ys);
		return PyObject_CallObject((PyObject*)&Document_type, Py_BuildValue("(Nnn)", self, ret.release(), 1));
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
		if (Py_TYPE(argDoc) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document type" };
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
