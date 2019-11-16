#include "../TopicModel/LLDA.h"

#include "module.h"

using namespace std;

static int LLDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01, sigma = 1, alphaEpsilon = 1e-10;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "rm_top", "k", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnffffn", (char**)kwlist, &tw, &minCnt, &rmTop,
		&K, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::ILLDAModel::create((tomoto::TermWeight)tw, K, alpha, eta, tomoto::RandGen{ seed });
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

static PyObject* LLDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr, *iter = nullptr, *iter2 = nullptr;
	static const char* kwlist[] = { "words", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argLabels)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ILLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		vector<string> labels;
		if(argLabels)
		{
			if (!(iter2 = PyObject_GetIter(argLabels)))
			{
				throw runtime_error{ "words must be an iterable of str." };
			}
			py::AutoReleaser arIter2{ iter2 };
			labels = py::makeIterToVector<string>(iter2);
		}
		auto ret = inst->addDoc(py::makeIterToVector<string>(iter), labels);
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

static PyObject* LLDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argLabels = nullptr, *iter = nullptr, *iter2 = nullptr;
	static const char* kwlist[] = { "words", "labels", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &argWords, &argLabels)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILLDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		vector<string> labels;
		if (argLabels)
		{
			if (!(iter2 = PyObject_GetIter(argLabels)))
			{
				throw runtime_error{ "words must be an iterable of str." };
			}
			py::AutoReleaser arIter2{ iter2 };
			labels = py::makeIterToVector<string>(iter2);
		}
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter), labels);
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

static PyObject* LLDA_getTopicLabelDict(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Dictionary_type, Py_BuildValue("(Nn)", self,
			&static_cast<tomoto::ILLDAModel*>(self->inst)->getTopicLabelDict()));
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

PyObject* Document_labels(DocumentObject* self, void* closure)
{
	auto makeReturn = [&](const tomoto::DocumentBase* doc, const Eigen::Matrix<int8_t, -1, 1>& labelMask)
	{
		auto inst = dynamic_cast<tomoto::ILLDAModel*>(self->parentModel->inst);
		vector<pair<string, float>> ret;
		auto topicDist = inst->getTopicsByDoc(doc);
		for (size_t i = 0; i < labelMask.size(); ++i)
		{
			if (labelMask[i])
			{
				ret.emplace_back(inst->getTopicLabelDict().toWord(i), topicDist[i]);
			}
		}
		return py::buildPyValue(ret);
	};

	try
	{
		if (!self->doc) throw runtime_error{ "doc is null!" };
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLLDA<tomoto::TermWeight::one>*>(self->doc);
			if (doc) return makeReturn(doc, doc->labelMask);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLLDA<tomoto::TermWeight::idf>*>(self->doc);
			if (doc) return makeReturn(doc, doc->labelMask);
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLLDA<tomoto::TermWeight::pmi>*>(self->doc);
			if (doc) return makeReturn(doc, doc->labelMask);
		} while (0);
		throw runtime_error{ "doc doesn't has 'labels' field!" };
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

DEFINE_LOADER(LLDA, LLDA_type);

static PyMethodDef LLDA_methods[] =
{
	{ "add_doc", (PyCFunction)LLDA_addDoc, METH_VARARGS | METH_KEYWORDS, LLDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)LLDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LLDA_make_doc__doc__ },
	{ "load", (PyCFunction)LLDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ nullptr }
};

static PyGetSetDef LLDA_getseters[] = {
	{ (char*)"topic_label_dict", (getter)LLDA_getTopicLabelDict, nullptr, LLDA_topic_label_dict__doc__, nullptr },
	{ nullptr },
};


PyTypeObject LLDA_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.LLDAModel",             /* tp_name */
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
	LLDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	LLDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	LLDA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)LLDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};
