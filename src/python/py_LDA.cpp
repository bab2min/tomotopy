#include <fstream>
#include <iostream>

#include "../TopicModel/LDA.h"

#include "module.h"

using namespace std;

static int LDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "rm_top", "k", "alpha", "eta", "seed", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnffn", (char**)kwlist, &tw, &minCnt, &rmTop, &K, &alpha, &eta, &seed)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, K, alpha, eta, tomoto::RandGen{ seed });
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

static PyObject* LDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->addDoc(py::makeIterToVector<string>(iter));
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

static PyObject* LDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *iter = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		py::AutoReleaser arIter{ iter };
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter));
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

static PyObject* LDA_train(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t iteration = 10, workers = 0;
	static const char* kwlist[] = { "iter", "workers", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nn", (char**)kwlist, &iteration, &workers)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
		inst->train(iteration, workers);
		Py_INCREF(Py_None);
		return Py_None;
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

static PyObject* LDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId, topN = 10;
	static const char* kwlist[] = { "topic_id", "top_n", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|n", (char**)kwlist, &topicId, &topN)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{"must topic_id < K"};
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
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

static PyObject* LDA_getTopicWordDist(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	size_t topicId;
	static const char* kwlist[] = { "topic_id", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n", (char**)kwlist, &topicId)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (topicId >= inst->getK()) throw runtime_error{ "must topic_id < K" };
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
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

static PyObject* LDA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argDoc, *iter = nullptr, *item;
	size_t iteration = 100, workers = 0, together = 0;
	float tolerance = -1;
	static const char* kwlist[] = { "doc", "iter", "tolerance", "workers", "together", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nfnp", (char**)kwlist, &argDoc, &iteration, &tolerance, &workers, &together)) return nullptr;
	DEBUG_LOG("infer " << self->ob_base.ob_type << ", " << self->ob_base.ob_refcnt);
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if ((iter = PyObject_GetIter(argDoc)) != nullptr)
		{
			py::AutoReleaser arIter{ iter };
			std::vector<tomoto::DocumentBase*> docs;
			while ((item = PyIter_Next(iter)))
			{
				py::AutoReleaser arItem{ item };
				if (Py_TYPE(item) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document type or list of tomotopy.Document" };
				auto* doc = (DocumentObject*)item;
				if (doc->parentModel != self) throw runtime_error{ "'doc' was from another model, not fit to this model" };
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
			}
			if (PyErr_Occurred()) throw bad_exception{};
			if (!self->isPrepared)
			{
				self->inst->prepare(true, self->minWordCnt, self->removeTopWord);
				self->isPrepared = true;
			}
			auto ll = self->inst->infer(docs, iteration, tolerance, workers, !!together);
			PyObject* ret = PyList_New(docs.size());
			size_t i = 0;
			for (auto d : docs)
			{
				PyList_SetItem(ret, i++, py::buildPyValue(self->inst->getTopicsByDoc(d)));
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
			if (Py_TYPE(argDoc) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document type or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)argDoc;
			if (doc->parentModel != self) throw runtime_error{ "'doc' was from another model, not fit to this model" };
			if (!self->isPrepared)
			{
				self->inst->prepare(true, self->minWordCnt, self->removeTopWord);
				self->isPrepared = true;
			}
			if (doc->owner)
			{
				std::vector<tomoto::DocumentBase*> docs;
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
				float ll = self->inst->infer(docs, iteration, tolerance, workers, !!together)[0];
				return Py_BuildValue("(Nf)", py::buildPyValue(self->inst->getTopicsByDoc(doc->doc)), ll);
			}
			else
			{
				return Py_BuildValue("(Ns)", py::buildPyValue(self->inst->getTopicsByDoc(doc->doc)), nullptr);
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

static PyObject* LDA_save(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	const char* filename;
	size_t full = 1;
	static const char* kwlist[] = { "filename", "full", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|p", (char**)kwlist, &filename, &full)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		ofstream str{ filename, ios_base::binary };
		if (!str) throw runtime_error{ std::string("cannot open file '") + filename + std::string("'") };
		self->inst->saveModel(str, !!full);
		Py_INCREF(Py_None);
		return Py_None;
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

static PyObject* LDA_getDocs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Corpus_type, Py_BuildValue("(O)", self));
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

static PyObject* LDA_getVocabs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return PyObject_CallObject((PyObject*)&Dictionary_type, Py_BuildValue("(Nn)", self, &self->inst->getVocabDict()));
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

static PyObject* LDA_getCountByTopics(TopicModelObject* self)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
		return py::buildPyValue(inst->getCountByTopic());
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

PyObject* LDA_getAlpha(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		vector<float> ret;
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			ret.emplace_back(inst->getAlpha(i));
		}
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

static PyObject* LDA_getRemovedTopWords(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->removeTopWord);
			self->isPrepared = true;
		}
		vector<string> ret;
		size_t last = inst->getVocabDict().size();
		for (size_t rmV = last - self->removeTopWord; rmV < last; ++rmV)
		{
			ret.emplace_back(inst->getVocabDict().toWord(rmV));
		}
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

DEFINE_GETTER(tomoto::ILDAModel, LDA, getK);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getEta);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getPerplexity);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getLLPerWord);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getTermWeight);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getN);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getV);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getVocabFrequencies);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getOptimInterval);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getBurnInIteration);

DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ILDAModel, LDA, setOptimInterval);
DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ILDAModel, LDA, setBurnInIteration);

DEFINE_LOADER(LDA, LDA_type);


PyObject* Document_LDA_Z(DocumentObject* self, void* closure)
{
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::one>*>(self->doc);
        if (doc) return py::buildPyValue(doc->Zs);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::idf>*>(self->doc);
        if (doc) return py::buildPyValue(doc->Zs);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::pmi>*>(self->doc);
        if (doc) return py::buildPyValue(doc->Zs);
    } while (0);
    return nullptr;
}


static PyMethodDef LDA_methods[] =
{
	{ "add_doc", (PyCFunction)LDA_addDoc, METH_VARARGS | METH_KEYWORDS, LDA_add_doc__doc__ },
	{ "make_doc", (PyCFunction)LDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LDA_make_doc__doc__},
	{ "train", (PyCFunction)LDA_train, METH_VARARGS | METH_KEYWORDS, LDA_train__doc__},
	{ "get_count_by_topics", (PyCFunction)LDA_getCountByTopics, METH_NOARGS, LDA_get_count_by_topics__doc__},
	{ "get_topic_words", (PyCFunction)LDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_words__doc__},
	{ "get_topic_word_dist", (PyCFunction)LDA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_word_dist__doc__ },
	{ "infer", (PyCFunction)LDA_infer, METH_VARARGS | METH_KEYWORDS, LDA_infer__doc__ },
	{ "save", (PyCFunction)LDA_save, METH_VARARGS | METH_KEYWORDS, LDA_save__doc__},
	{ "load", (PyCFunction)LDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__},
	{ nullptr }
};

static PyGetSetDef LDA_getseters[] = {
	{ (char*)"tw", (getter)LDA_getTermWeight, nullptr, LDA_tw__doc__, nullptr },
	{ (char*)"perplexity", (getter)LDA_getPerplexity, nullptr, LDA_perplexity__doc__, nullptr },
	{ (char*)"ll_per_word", (getter)LDA_getLLPerWord, nullptr, LDA_ll_per_word__doc__, nullptr },
	{ (char*)"k", (getter)LDA_getK, nullptr, LDA_k__doc__, nullptr },
	{ (char*)"alpha", (getter)LDA_getAlpha, nullptr, LDA_alpha__doc__, nullptr },
	{ (char*)"eta", (getter)LDA_getEta, nullptr, LDA_eta__doc__, nullptr },
	{ (char*)"docs", (getter)LDA_getDocs, nullptr, LDA_docs__doc__, nullptr },
	{ (char*)"vocabs", (getter)LDA_getVocabs, nullptr, LDA_vocabs__doc__, nullptr },
	{ (char*)"num_vocabs", (getter)LDA_getV, nullptr, LDA_num_vocabs__doc__, nullptr },
	{ (char*)"vocab_freq", (getter)LDA_getVocabFrequencies, nullptr, LDA_vocab_freq__doc__, nullptr },
	{ (char*)"num_words", (getter)LDA_getN, nullptr, LDA_num_words__doc__, nullptr },
	{ (char*)"optim_interval", (getter)LDA_getOptimInterval, (setter)LDA_setOptimInterval, LDA_optim_interval__doc__, nullptr },
	{ (char*)"burn_in", (getter)LDA_getBurnInIteration, (setter)LDA_setBurnInIteration, LDA_burn_in__doc__, nullptr },
	{ (char*)"removed_top_words", (getter)LDA_getRemovedTopWords, nullptr, LDA_removed_top_words__doc__, nullptr },
	{ nullptr },
};

PyTypeObject LDA_type = {
	PyVarObject_HEAD_INIT(nullptr, 0)
	"tomotopy.LDAModel",             /* tp_name */
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
	LDA___init____doc__,           /* tp_doc */
	0,                         /* tp_traverse */
	0,                         /* tp_clear */
	0,                         /* tp_richcompare */
	0,                         /* tp_weaklistoffset */
	0,                         /* tp_iter */
	0,                         /* tp_iternext */
	LDA_methods,             /* tp_methods */
	0,						 /* tp_members */
	LDA_getseters,                         /* tp_getset */
	0,                         /* tp_base */
	0,                         /* tp_dict */
	0,                         /* tp_descr_get */
	0,                         /* tp_descr_set */
	0,                         /* tp_dictoffset */
	(initproc)LDA_init,      /* tp_init */
	PyType_GenericAlloc,
	PyType_GenericNew,
};
