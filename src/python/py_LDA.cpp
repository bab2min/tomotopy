#include <fstream>
#include <iostream>

#include "../TopicModel/LDA.h"

#include "utils.h"
#include "module.h"

using namespace std;

static int LDA_init(TopicModelObject *self, PyObject *args, PyObject *kwargs)
{
	size_t tw = 0, minCnt = 0, minDf = 0, rmTop = 0;
	size_t K = 1;
	float alpha = 0.1f, eta = 0.01f;
	PyObject* objCorpus = nullptr, *objTransform = nullptr;
	size_t seed = random_device{}();
	static const char* kwlist[] = { "tw", "min_cf", "min_df", "rm_top", "k", "alpha", "eta", "seed",
		"corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|nnnnnffnOO", (char**)kwlist, 
		&tw, &minCnt, &minDf, &rmTop, &K, &alpha, &eta, &seed, &objCorpus, &objTransform)) return -1;
	try
	{
		tomoto::ITopicModel* inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, K, alpha, eta, seed);
		if (!inst) throw runtime_error{ "unknown tw value" };
		self->inst = inst;
		self->isPrepared = false;
		self->minWordCnt = minCnt;
		self->minWordDf = minDf;
		self->removeTopWord = rmTop;
		self->initParams = py::buildPyDict(kwlist,
			tw, minCnt, minDf, rmTop, K, alpha, eta, seed
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
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] `words` should be an iterable of str.");
		tomoto::RawDoc raw = buildRawDoc(argWords);
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

static PyObject* LDA_addCorpus(TopicModelObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject* corpus, *transform = nullptr;
	static const char* kwlist[] = { "corpus", "transform", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", (char**)kwlist, &corpus, &transform)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		if (self->isPrepared) throw runtime_error{ "cannot add_corpus() after train()" };
		if (!PyObject_TypeCheck(corpus, &UtilsCorpus_type)) throw runtime_error{ "`corpus` must be an instance of `tomotopy.utils.Corpus`" };
		py::UniqueObj _corpusRet{ PyObject_CallFunctionObjArgs((PyObject*)&UtilsCorpus_type, (PyObject*)self, nullptr) };
		CorpusObject* corpusRet = (CorpusObject*)_corpusRet.get();
		corpusRet->docIdcs = insertCorpus(self, corpus, transform);
		for (size_t i = 0; i < corpusRet->docIdcs.size(); ++i)
		{
			corpusRet->invmap.emplace(self->inst->getDoc(corpusRet->docIdcs[i])->docUid, i);
		}
		return _corpusRet.release();
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

static DocumentObject* LDA_makeDoc(TopicModelObject* self, PyObject* args, PyObject *kwargs)
{
	PyObject *argWords = nullptr;
	static const char* kwlist[] = { "words", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", (char**)kwlist, &argWords)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* inst = static_cast<tomoto::ILDAModel*>(self->inst);
		if (PyUnicode_Check(argWords)) PRINT_WARN_ONCE("[warn] 'words' should be an iterable of str.");
		tomoto::RawDoc raw = buildRawDoc(argWords);
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
		inst->setWordPrior(word, py::toCpp<vector<tomoto::Float>>(prior, "`prior` must be a list of floats with len = k"));
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

PyObject* LDA_infer(TopicModelObject* self, PyObject* args, PyObject *kwargs)
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
				docs.emplace_back((tomoto::DocumentBase*)doc->doc);
			}
			if (PyErr_Occurred()) throw bad_exception{};
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
			if (!PyObject_TypeCheck(argDoc, &UtilsDocument_type)) throw runtime_error{ "`doc` must be tomotopy.Document type or list of tomotopy.Document" };
			auto* doc = (DocumentObject*)argDoc;
			if (doc->corpus->tm != self) throw runtime_error{ "`doc` was from another model, not fit to this model" };
			if (doc->owner)
			{
				std::vector<tomoto::DocumentBase*> docs;
				docs.emplace_back((tomoto::DocumentBase*)doc->getBoundDoc());
				float ll = self->inst->infer(docs, iteration, tolerance, workers, (tomoto::ParallelScheme)ps, !!together)[0];
				return Py_BuildValue("(Nf)", py::buildPyValue(self->inst->getTopicsByDoc(doc->getBoundDoc())), ll);
			}
			else
			{
				return Py_BuildValue("(Ns)", py::buildPyValue(self->inst->getTopicsByDoc(doc->getBoundDoc())), nullptr);
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

		vector<uint8_t> extra_data;
		{
			py::UniqueObj pickle{ PyImport_ImportModule("pickle") };
			PyObject* pickle_dict{ PyModule_GetDict(pickle) };
			py::UniqueObj args{ Py_BuildValue("(O)", self->initParams) };
			py::UniqueObj pickled_bytes{ PyObject_CallObject(
				PyDict_GetItemString(pickle_dict, "dumps"),
				args
			) };
			char* buf;
			ssize_t bufsize;
			PyBytes_AsStringAndSize(pickled_bytes, &buf, &bufsize);
			extra_data.resize(bufsize);
			memcpy(extra_data.data(), buf, bufsize);
		}

		self->inst->saveModel(str, !!full, &extra_data);
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
		self->inst->updateVocab(py::toCpp<vector<string>>(objWords, "`words` must be an iterable of str"));
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
static CorpusObject* LDA_getDocs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		py::UniqueObj args{ py::buildPyTuple((PyObject*)self) };
		auto ret = (CorpusObject*)PyObject_CallObject((PyObject*)&UtilsCorpus_type, args);
		return ret;
	}
	catch (const bad_exception&)
	{
	}
	catch (const exception& e)
	{
		PyErr_SetString(PyExc_Exception, e.what());
	}
	return nullptr;
}

static VocabObject* LDA_getVocabs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&self->inst->getVocabDict();
		ret->size = self->inst->getVocabDict().size();
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

static VocabObject* LDA_getUsedVocabs(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		auto* ret = (VocabObject*)PyObject_CallObject((PyObject*)&UtilsVocab_type, nullptr);
		ret->dep = (PyObject*)self;
		Py_INCREF(ret->dep);
		ret->vocabs = (tomoto::Dictionary*)&self->inst->getVocabDict();
		ret->size = self->inst->getV();
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

static PyObject* LDA_getUsedVocabCf(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->getVocabCf().begin(), self->inst->getVocabCf().begin() + self->inst->getV());
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

static PyObject* LDA_getUsedVocabDf(TopicModelObject* self, void* closure)
{
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };
		return py::buildPyValue(self->inst->getVocabDf().begin(), self->inst->getVocabDf().begin() + self->inst->getV());
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
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
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
		/*if (!self->isPrepared)
		{
			inst->prepare(true, self->minWordCnt, self->minWordDf, self->removeTopWord);
			self->isPrepared = true;
		}*/
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

static PyObject* LDA_summary(TopicModelObject* self, PyObject* args, PyObject* kwargs)
{
	PyObject *argInitialHP = nullptr, 
		*argParams = nullptr,
		*argTopicWordTopN = nullptr,
		*argFile = nullptr, 
		*argFlush = nullptr;
	static const char* kwlist[] = { "initial_hp", "params", "topic_word_top_n", "file", "flush", nullptr };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OOOOO", (char**)kwlist,
		&argInitialHP, &argParams, &argTopicWordTopN, &argFile, &argFlush)) return nullptr;
	try
	{
		if (!self->inst) throw runtime_error{ "inst is null" };

		py::UniqueObj mod{ PyImport_ImportModule("tomotopy._summary") };
		if (!mod) throw bad_exception{};
		PyObject* mod_dict = PyModule_GetDict(mod);
		if (!mod_dict) throw bad_exception{};
		PyObject* summary_func = PyDict_GetItemString(mod_dict, "summary");
		if (!summary_func) throw bad_exception{};
		py::UniqueObj args{ Py_BuildValue("(O)", self) };
		py::UniqueObj kwargs{ py::buildPyDictSkipNull(kwlist,
			argInitialHP, argParams, argTopicWordTopN,
			argFile, argFlush
		) };
		return PyObject_Call(summary_func, args, kwargs);
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
DEFINE_GETTER(tomoto::ILDAModel, LDA, getVocabDf);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getOptimInterval);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getBurnInIteration);
DEFINE_GETTER(tomoto::ILDAModel, LDA, getGlobalStep);

DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ILDAModel, LDA, setOptimInterval);
DEFINE_SETTER_NON_NEGATIVE_INT(tomoto::ILDAModel, LDA, setBurnInIteration);

DEFINE_LOADER(LDA, LDA_type);


PyObject* Document_LDA_Z(DocumentObject* self, void* closure)
{
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::one>*>(self->getBoundDoc());
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::idf>*>(self->getBoundDoc());
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    do
    {
        auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::pmi>*>(self->getBoundDoc());
        if (doc) return buildPyValueReorder(doc->Zs, doc->wOrder);
    } while (0);
    return nullptr;
}

PyObject* Document_getCountVector(DocumentObject* self)
{
	try
	{
		if (self->corpus->isIndependent()) throw runtime_error{ "This method can only be called by documents bound to the topic model." };
		if (!self->corpus->tm->inst) throw runtime_error{ "inst is null" };
		size_t v = self->corpus->tm->inst->getV();
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::one>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(doc->getCountVector(v));
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::idf>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(doc->getCountVector(v));
		} while (0);
		do
		{
			auto* doc = dynamic_cast<const tomoto::DocumentLDA<tomoto::TermWeight::pmi>*>(self->getBoundDoc());
			if (doc) return py::buildPyValue(doc->getCountVector(v));
		} while (0);
		
		throw runtime_error{ "cannot get count vector" };
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

PyObject* LDA_getInitParams(TopicModelObject* self)
{
	return PyDict_Copy(self->initParams);
}

static PyMethodDef LDA_methods[] =
{
	{ "add_doc", (PyCFunction)LDA_addDoc, METH_VARARGS | METH_KEYWORDS, LDA_add_doc__doc__ },
	{ "add_corpus", (PyCFunction)LDA_addCorpus, METH_VARARGS | METH_KEYWORDS, LDA_add_corpus__doc__ },
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
	{ "summary", (PyCFunction)LDA_summary, METH_VARARGS | METH_KEYWORDS, LDA_summary__doc__},
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
	{ (char*)"used_vocabs", (getter)LDA_getUsedVocabs, nullptr, LDA_used_vocabs__doc__, nullptr },
	{ (char*)"used_vocab_freq", (getter)LDA_getUsedVocabCf, nullptr, LDA_used_vocab_freq__doc__, nullptr },
	{ (char*)"vocab_df", (getter)LDA_getVocabDf, nullptr, LDA_vocab_df__doc__, nullptr },
	{ (char*)"used_vocab_df", (getter)LDA_getUsedVocabDf, nullptr, LDA_used_vocab_df__doc__, nullptr },
	{ (char*)"global_step", (getter)LDA_getGlobalStep, nullptr, LDA_global_step__doc__, nullptr },
	{ (char*)"_init_params", (getter)LDA_getInitParams, nullptr, "", nullptr },
	{ nullptr },
};

TopicModelTypeObject LDA_type = { {
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
} };
