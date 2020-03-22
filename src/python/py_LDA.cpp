#include <fstream>
#include <iostream>

#include "../TopicModel/LDA.h"

#include "module.h"

using namespace std;

static int LDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1, eta = 0.01;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "alpha", "eta", "seed", 
		"corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnffnOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &K, &alpha, &eta, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		if (objCorpus && !PyObject_HasAttrString(objCorpus, corpus_feeder_name))
		{
			throw runtime_error{ "`corpus` must be `tomotopy.utils.Corpus` type." };
		}

		tomoto::ITopicModel* inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, K, alpha, eta, tomoto::RandGen{ seed });
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

static PyObject* LDA_addDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_doc() after train()" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
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

static PyObject* LDA_addDoc_(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords, *argStartPos = nullptr, *argLength = nullptr;
	const char* argRaw = nullptr;
	static const char* kwlist[] = { "words", "raw", "start_pos", "length", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|sOO", (char**)kwlist, 
		&argWords, &argRaw, &argStartPos, &argLength)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		string raw;
		if (argRaw) raw = argRaw;

		py::UniqueObj iter = PyObject_GetIter(argWords);
		vector<tomoto::Vid> words = py::makeIterToVector<tomoto::Vid>(iter);
		iter = PyObject_GetIter(argStartPos);
		vector<uint32_t> startPos = py::makeIterToVector<uint32_t>(iter);
		iter = PyObject_GetIter(argLength);
		vector<uint16_t> length = py::makeIterToVector<uint16_t>(iter);
		char2Byte(raw, startPos, length);
		auto ret = inst->addDoc(raw, words, startPos, length);
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
	PyObject *argWords = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN("[warn] 'words' should be an iterable of str.");
		py::UniqueObj iter;
		if (!(iter = PyObject_GetIter(argWords)))
		{
			throw runtime_error{ "words must be an iterable of str." };
		}
		auto ret = inst->makeDoc(py::makeIterToVector<string>(iter));
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

PyObject* LDA_setWordPrior(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	const char* word;
	PyObject* prior;
	static const char* kwlist[] = { "word", "prior", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "sO", (char**)kwlist, &word, &prior)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot set_word_prior() after train()" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		py::UniqueObj iter = PyObject_GetIter(prior);
		if (!iter) throw runtime_error{ "`prior` must be a list of floats with len = k" };
		inst->setWordPrior(word, py::makeIterToVector<float>(iter));
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

PyObject* LDA_getWordPrior(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	const char* word;
	static const char* kwlist[] = { "word", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s", (char**)kwlist, &word)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		return py::buildPyValue(inst->getWordPrior(word));
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
	size_t iteration = 10, workers = 0, ps = 0;
	static const char* kwlist[] = { "iter", "workers", "parallel", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnn", (char**)kwlist, &iteration, &workers, &ps)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}
		inst->train(iteration, workers, (tomoto::ParallelScheme)ps);
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

PyObject* LDA_getTopicWords(TopicModelObject* self, PyObject* args, PyObject *kwargs)
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
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
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
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
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

PyObject* LDA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argDoc;
	size_t iteration = 100, workers = 0, together = 0, ps = 0;
	float tolerance = -1;
	static const char* kwlist[] = { "doc", "iter", "tolerance", "workers", "parallel", "together", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|nfnnp", (char**)kwlist, &argDoc, &iteration, &tolerance, &workers, &ps, &together)) return nullptr;
	DEBUG_LOG("infer " << self->ob_base.ob_type << ", " << self->ob_base.ob_refcnt);
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		py::UniqueObj iter;
		if ((iter = PyObject_GetIter(argDoc)) != nullptr)
		{
			std::vector<tomoto::DocumentBase*> docs;
			py::UniqueObj item;
			while ((item = PyIter_Next(iter)))
			{
				if (Py_TYPE(item) != &Document_type) throw runtime_error{ "'doc' must be tomotopy.Document type or list of tomotopy.Document" };
				auto* doc = (DocumentObject*)item.get();
				if (doc->parentModel != self) throw runtime_error{ "'doc' was from another model, not fit to this model" };
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
			}
			if (PyErr_Occurred()) throw bad_exception{};
			if (!self->isPrepared)
			{
				self->inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
				self->isPrepared = true;
			}
			auto ll = self->inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together);
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
				self->inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
				self->isPrepared = true;
			}
			if (doc->owner)
			{
				std::vector<tomoto::DocumentBase*> docs;
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
				float ll = self->inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together)[0];
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

static PyObject* LDA_update_vocab(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject* objWords;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &objWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		py::UniqueObj iter = PyObject_GetIter(objWords);
		self->inst->updateVocab(py::makeIterToVector<string>(iter));
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
		py::UniqueObj args = Py_BuildValue("(O)", self);
		return PyObject_CallObject((PyObject*)&Corpus_type, args);
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
		py::UniqueObj args = Py_BuildValue("(On)", self, &self->inst->getVocabDict());
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

static PyObject* LDA_getCountByTopics(TopicModelObject* self)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
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
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
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
DEFINE_GETTER(tomoto::ILDAModel, LDA, getVocabCf);
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
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::idf>*>(self->doc);
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::pmi>*>(self->doc);
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    return nullptr;
}


static PyMethodDef LDA_methods[] =
{
	{ "add_doc", (PyCFunction)LDA_addDoc, METH_VARARGS | METH_KEYWORDS, LDA_add_doc__doc__ },
	{ "_add_doc", (PyCFunction)LDA_addDoc_, METH_VARARGS | METH_KEYWORDS, "" },
	{ "make_doc", (PyCFunction)LDA_makeDoc, METH_VARARGS | METH_KEYWORDS, LDA_make_doc__doc__},
	{ "set_word_prior", (PyCFunction)LDA_setWordPrior, METH_VARARGS | METH_KEYWORDS, LDA_set_word_prior__doc__},
	{ "get_word_prior", (PyCFunction)LDA_getWordPrior, METH_VARARGS | METH_KEYWORDS, LDA_get_word_prior__doc__},
	{ "train", (PyCFunction)LDA_train, METH_VARARGS | METH_KEYWORDS, LDA_train__doc__},
	{ "get_count_by_topics", (PyCFunction)LDA_getCountByTopics, METH_NOARGS, LDA_get_count_by_topics__doc__},
	{ "get_topic_words", (PyCFunction)LDA_getTopicWords, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_words__doc__},
	{ "get_topic_word_dist", (PyCFunction)LDA_getTopicWordDist, METH_VARARGS | METH_KEYWORDS, LDA_get_topic_word_dist__doc__ },
	{ "infer", (PyCFunction)LDA_infer, METH_VARARGS | METH_KEYWORDS, LDA_infer__doc__ },
	{ "save", (PyCFunction)LDA_save, METH_VARARGS | METH_KEYWORDS, LDA_save__doc__},
	{ "load", (PyCFunction)LDA_load, METH_STATIC | METH_VARARGS | METH_KEYWORDS, LDA_load__doc__},
	{ "_update_vocab", (PyCFunction)LDA_update_vocab, METH_VARARGS | METH_KEYWORDS, ""},
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
	{ (char*)"vocab_freq", (getter)LDA_getVocabCf, nullptr, LDA_vocab_freq__doc__, nullptr },
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
