#include "../TopicModel/PA.h"

#include "module.h"
#include "utils.h"

using namespace std;

static int PA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1, K2 = 1;
	float alpha = 0.1f, eta = 0.01f;
	size_t seed = random_device{}();
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k1", "k2", "alpha", "eta", 
		"seed", "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnnffnOO", (char**)kwlist, &tw, &minCnt, &minDf, &rmTop,
		&K, &K2, &alpha, &eta, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::IPAModel::create((tomoto::TermWeight)tw, 
			K, K2, alpha, eta, seed);
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, K, K2, alpha, eta, seed
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

static PyObject* PA_getSubTopicDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "super_topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k1" };
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getSubTopicBySuperTopic(topicId));
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

static PyObject* PA_getSubTopics(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "super_topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < k1" };
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getSubTopicBySuperTopicSorted(topicId, topN));

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

static PyObject* PA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "sub_topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK2()) throw runtime_error{ "must topic_id < k2" };
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
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

static PyObject* PA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "sub_topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		if (topicId >= inst->getK2()) throw runtime_error{ "must topic_id < k2" };
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getWidsByTopic(topicId));
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


PyObject* Document_getSubTopics(DocumentObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topN = 10;
	static const char* kwlist[] = { "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|n", (char**)kwlist, &topN)) return nullptr;
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->corpus->tm->inst);
		if (!self->corpus->tm->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(inst->getSubTopicsByDocSorted(self->getBoundDoc(), topN));
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

PyObject* Document_getSubTopicDist(DocumentObject* self)
{
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->corpus->tm->inst);
		if (!self->corpus->tm->isPrepared) throw runtime_error{ "train() should be called first for calculating the topic distribution" };
		return py::buildPyValue(inst->getSubTopicsByDoc(self->getBoundDoc()));
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

static PyObject* PA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argDoc, *argTransform = nullptr;
	size_t iteration = 100, workers = 0, together = 0, ps = 0;
	float tolerance = -1;
	static const char* kwlist[] = { "doc", "iter", "tolerance", "workers", "parallel", "together", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nfnnpO", (char**)kwlist, &argDoc, &iteration, &tolerance, &workers, &ps, &together, &argTransform)) return nullptr;
	DEBUG_LOG("infer " << self->ob_base.ob_type << ", " << self->ob_base.ob_refcnt);
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (!self->isPrepared) throw runtime_error{ "cannot infer with untrained model" };
		auto inst = static_cast<tomoto::IPAModel*>(self->inst);
		py::UniqueObj iter;
		if (PyObject_TypeCheck(argDoc, &UtilsCorpus_type))
		{
			CorpusObject* cps = makeCorpus(self, argDoc, argTransform);
			std::vector<tomoto::DocumentBase*> docs;
			for (auto& d : cps->docsMade) docs.emplace_back(d.get());
			auto ll = self->inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together);
			return py::buildPyTuple(py::UniqueObj{ (PyObject*)cps }, ll);
		}
		else if ((iter = py::UniqueObj{ PyObject_GetIter(argDoc) }) != nullptr)
		{
			std::vector<tomoto::DocumentBase*> docs;
			py::UniqueObj item;
			while ((item = py::UniqueObj{ PyIter_Next(iter) }))
			{
				if (!PyObject_TypeCheck(item, &UtilsDocument_type)) throw runtime_error{ "`doc` must be tomotopy.Document type or list of tomotopy.Document" };
				auto* doc = (DocumentObject*)item.get();
				if (doc->corpus->tm != self) throw runtime_error{ "`doc` was from another model, not fit to this model" };
				docs.emplace_back((tomoto::DocumentBase*)doc->getBoundDoc());
			}
			if (PyErr_Occurred()) throw bad_exception{};
			if (!self->isPrepared) throw runtime_error{ "cannot infer with untrained model" };
			auto ll = inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together);
			PyObject* ret = PyList_New(docs.size());
			size_t i = 0;
			for (auto d : docs)
			{
				PyList_SetItem(ret, i++, Py_BuildValue("(NN)", 
					py::buildPyValue(inst->getTopicsByDoc(d)),
					py::buildPyValue(inst->getSubTopicsByDoc(d))
				));
			}
			if (together)
			{
				return Py_BuildValue("(Nf)", ret, ll[0]);
			}
			else
			{
				return Py_BuildValue("(NN)", ret, py::buildPyValue(ll));
			}
		}
		else
		{
			PyErr_Clear();
			if (!PyObject_TypeCheck(argDoc, &UtilsDocument_type)) throw runtime_error{ "'doc' must be tomotopy.Document type or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)argDoc;
			if (doc->corpus->tm != self) throw runtime_error{ "'doc' was from another model, not fit to this model" };
			if (doc->owner)
			{
				std::vector<tomoto::DocumentBase*> docs;
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
				double ll = self->inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together)[0];
				return Py_BuildValue("((NN)f)", py::buildPyValue(inst->getTopicsByDoc(doc->getBoundDoc())), 
					py::buildPyValue(inst->getSubTopicsByDoc(doc->getBoundDoc())), ll);
			}
			else
			{
				return Py_BuildValue("((NN)s)", py::buildPyValue(inst->getTopicsByDoc(doc->getBoundDoc())),
					py::buildPyValue(inst->getSubTopicsByDoc(doc->getBoundDoc())), nullptr);
			}
		}
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

static PyObject* PA_getCountBySuperTopic(TopicModelObject* self)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
		return py::buildPyValue(inst->getCountBySuperTopic());
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

DEFINE_GETTER(tomoto::IPAModel, PA, getK2);
DEFINE_DOCUMENT_GETTER_REORDER(tomoto::DocumentPA, Z2, Z2s);
DEFINE_LOADER(PA, PA_type);

static PyMethodDef PA_methods[] =
{
	{ "load", (PyCFunction)PA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__ },
	{ "get_sub_topic_dist", (PyCFunction)PA_getSubTopicDist, METH_VARARGS | METH_KEYWORDS, PA_get_sub_topic_dist__doc__ },
	{ "get_sub_topics", (PyCFunction)PA_getSubTopics, METH_VARARGS | METH_KEYWORDS, PA_get_sub_topics__doc__ },
	{ "get_topic_words", (PyCFunction)PA_getTopicWords, METH_VARARGS | METH_KEYWORDS, PA_get_topic_words__doc__},
	{ "get_topic_word_dist", (PyCFunction)PA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, PA_get_topic_word_dist__doc__ },
	{ "get_count_by_super_topic", (PyCFunction)PA_getCountBySuperTopic, METH_VARARGS | METH_KEYWORDS, PA_get_count_by_super_topic__doc__ },
	{ "infer", (PyCFunction)PA_infer, METH_VARARGS | METH_KEYWORDS, PA_infer__doc__ },
	{ nullptr }
};

static PyObject* PA_getSubalpha(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::IPAModel*>(self->inst);
		npy_intp shapes[2] = { (npy_intp)inst->getK(), (npy_intp)inst->getK2() };
		PyObject* ret = PyArray_EMPTY(2, shapes, NPY_FLOAT, 0);
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getSubAlpha(i);
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

static PyGetSetDef PA_getseters[] = {
	{ (char*)"k1", (getter)LDA_getK, nullptr, PA_k1__doc__, nullptr },
	{ (char*)"k2", (getter)PA_getK2, nullptr, PA_k2__doc__, nullptr },
	{ (char*)"alpha", (getter)LDA_getAlpha, nullptr, PA_alpha__doc__, nullptr },
	{ (char*)"subalpha", (getter)PA_getSubalpha, nullptr, PA_subalpha__doc__, nullptr },
	{ nullptr },
};

TopicModelTypeObject PA_type = { {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.PAModel",             /* tp_name */
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
	PA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	PA_methods,             /* tp_methods */
	0,						 /* tp_members */
	PA_getseters,                         /* tp_getset */
	&LDA_type,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)PA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
} };
