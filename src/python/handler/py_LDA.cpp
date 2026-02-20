#include <fstream>
#include <iostream>
#include <sstream>

#include "../../TopicModel/LDA.h"

#include "utils.h"
#include "module.h"

using namespace std;

inline tomoto::RawDoc::MiscType transformMisc(const tomoto::RawDoc::MiscType& misc, PyObject* transform)
{
	if (!transform || transform == Py_None) return misc;
	py::UniqueObj args{ py::buildPyValue(misc) };
	py::UniqueObj res{ PyObject_CallFunctionObjArgs(transform, args.get(), nullptr) };
	if (!res) throw py::ExcPropagation{};
	tomoto::RawDoc::MiscType ret;
	if (!py::toCpp(res.get(), ret))
	{
		throw py::ValueError{ "`transform` must return an instance of `dict`." };
	}
	return ret;
}

tomoto::RawDoc::MiscType LDAModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	return {};
}

std::vector<size_t> LDAModelObject::insertCorpus(PyObject* corpusObj, PyObject* transform)
{
	vector<size_t> ret;
	if (!corpusObj || corpusObj == Py_None) return ret;
	if (!PyObject_TypeCheck(corpusObj, py::Type<CorpusObject>)) throw py::ValueError{ "`corpus` must be an instance of `tomotopy.utils.Corpus`" };
	auto corpus = (CorpusObject*)corpusObj;
	bool insert_into_empty = inst->updateVocab(corpus->getVocabDict().getRaw());
	if (corpus->isIndependent())
	{
		for (auto& rdoc : corpus->docs)
		{
			tomoto::RawDoc doc;
			doc.rawStr = rdoc.rawStr;
			doc.weight = rdoc.weight;
			doc.docUid = rdoc.docUid;

			for (size_t i = 0; i < rdoc.words.size(); ++i)
			{
				if (rdoc.words[i] == tomoto::non_vocab_id) continue;

				if (insert_into_empty) doc.words.emplace_back(rdoc.words[i]);
				else doc.words.emplace_back(corpus->getVocabDict().mapToNewDict(rdoc.words[i], inst->getVocabDict()));

				if (!doc.rawStr.empty())
				{
					doc.origWordPos.emplace_back(rdoc.origWordPos[i]);
					doc.origWordLen.emplace_back(rdoc.origWordLen[i]);
				}
			}

			if (doc.words.empty())
			{
				fprintf(stderr, "Adding empty document was ignored.\n");
				continue;
			}

			if (!doc.rawStr.empty()) char2Byte(doc.rawStr, doc.origWordPos, doc.origWordLen);
			doc.misc = convertMisc(transformMisc(rdoc.misc, transform));
			ret.emplace_back(inst->addDoc(doc));
		}
	}
	else
	{
		for (Py_ssize_t i = 0; i < corpus->len(); ++i)
		{
			auto& rdoc = (tomoto::DocumentBase&)*corpus->getDoc(i);

			tomoto::RawDoc doc;
			doc.rawStr = rdoc.rawStr;
			doc.weight = rdoc.weight;
			doc.docUid = rdoc.docUid;

			for (size_t i = 0; i < rdoc.words.size(); ++i)
			{
				if (rdoc.words[i] == tomoto::non_vocab_id) continue;

				doc.words.emplace_back(corpus->getVocabDict().mapToNewDict(rdoc.words[i], inst->getVocabDict()));

				if (!doc.rawStr.empty())
				{
					doc.origWordPos.emplace_back(rdoc.origWordPos[i]);
					doc.origWordLen.emplace_back(rdoc.origWordLen[i]);
				}
			}

			if (doc.words.empty())
			{
				fprintf(stderr, "Adding empty document was ignored.\n");
				continue;
			}

			if (!doc.rawStr.empty()) char2Byte(doc.rawStr, doc.origWordPos, doc.origWordLen);
			doc.misc = convertMisc(transformMisc(rdoc.makeMisc(corpus->tm->inst.get()), transform));
			ret.emplace_back(inst->addDoc(doc));
		}

	}
	return ret;
}

py::UniqueCObj<CorpusObject> LDAModelObject::makeCorpus(PyObject* _corpus, PyObject* transform) const
{
	if (!_corpus || _corpus == Py_None) return {};
	auto* corpus = py::checkType<CorpusObject>(_corpus, "`corpus` must be an instance of `tomotopy.utils.Corpus`");
	py::UniqueCObj<CorpusObject> corpusMade{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr)};
	corpusMade->made = true;
	for (auto& rdoc : corpus->docs)
	{
		tomoto::RawDoc doc;
		doc.rawStr = rdoc.rawStr;
		doc.weight = rdoc.weight;
		doc.docUid = rdoc.docUid;

		for (size_t i = 0; i < rdoc.words.size(); ++i)
		{
			if (rdoc.words[i] == tomoto::non_vocab_id) continue;
			tomoto::Vid w = corpus->getVocabDict().mapToNewDict(rdoc.words[i], inst->getVocabDict());
			if (w == tomoto::non_vocab_id) continue;
			doc.words.emplace_back(w);

			if (!doc.rawStr.empty())
			{
				doc.origWordPos.emplace_back(rdoc.origWordPos[i]);
				doc.origWordLen.emplace_back(rdoc.origWordLen[i]);
			}
		}

		if (doc.words.empty())
		{
			fprintf(stderr, "Adding empty document was ignored.\n");
			continue;
		}

		if (!doc.rawStr.empty()) char2Byte(doc.rawStr, doc.origWordPos, doc.origWordLen);
		doc.misc = convertMisc(transformMisc(rdoc.misc, transform));
		corpusMade->docsMade.emplace_back(inst->makeDoc(doc));
	}
	return corpusMade;
}

LDAModelObject::LDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, PyObject* alpha, float eta, PyObject* seed,
	PyObject* corpus, PyObject* transform)
{
	tomoto::LDAArgs mArgs;
	mArgs.k = k;
	if (alpha)
	{
		mArgs.alpha = broadcastObj<tomoto::Float>(alpha, mArgs.k,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
		);
	}
	mArgs.eta = eta;
	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, mArgs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, mArgs);
	if (!inst) throw py::ValueError{ "unknown tw value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

py::UniqueObj LDAModelObject::addCorpus(PyObject* corpus, PyObject* transform)
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (isPrepared) throw py::RuntimeError{ "cannot add_corpus() after train()" };
	if (!PyObject_TypeCheck(corpus, py::Type<CorpusObject>)) throw py::ValueError{ "`corpus` must be an instance of `tomotopy.utils.Corpus`" };
	py::UniqueObj _corpusRet{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr)};
	CorpusObject* corpusRet = (CorpusObject*)_corpusRet.get();
	corpusRet->docIdcs = insertCorpus(corpus, transform);
	for (size_t i = 0; i < corpusRet->docIdcs.size(); ++i)
	{
		corpusRet->invmap.emplace(inst->getDoc(corpusRet->docIdcs[i])->docUid, i);
	}
	return _corpusRet;
}

std::optional<size_t> LDAModelObject::addDoc(PyObject* words, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::ILDAModel>();
	if (PyUnicode_Check(words))
	{
		throw py::ValueError{ "`words` should be an iterable of str." };
	}
		
	tomoto::RawDoc raw = buildRawDoc(words);
	try
	{
		auto ret = inst->addDoc(raw);
		return py::buildPyValue(ret);
	}
	catch (const tomoto::exc::EmptyWordArgument&)
	{
		if (ignoreEmptyWords)
		{
			return std::nullopt;
		}
		else
		{
			throw;
		}
	}
}

py::UniqueCObj<DocumentObject> LDAModelObject::makeDoc(PyObject* words)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::ILDAModel>();
	if (PyUnicode_Check(words))
	{
		throw py::ValueError{ "`words` should be an iterable of str." };
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	auto doc = inst->makeDoc(raw);
	py::UniqueCObj<CorpusObject> corpus{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr) };
	auto ret = py::makeNewObject<DocumentObject>(getDocumentCls());
	ret->corpus = corpus.copy();
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

void LDAModelObject::setWordPrior(const std::string& word, PyObject* prior)
{
	if (isPrepared) throw py::RuntimeError{ "cannot set_word_prior() after train()" };
	auto* inst = getInst<tomoto::ILDAModel>();
	if (PyDict_Check(prior))
	{
		vector<tomoto::Float> priors(inst->getNumTopicsForPrior(), inst->getEta());
		PyObject* key, * value;
		Py_ssize_t pos = 0;
		while (PyDict_Next(prior, &pos, &key, &value))
		{
			auto k = PyLong_AsLong(key);
			if (k < 0 || k >= priors.size()) throw py::ValueError{ "`prior` must be a dict of {topic_id: float}" };
			auto v = PyFloat_AsDouble(value);
			if (PyErr_Occurred()) throw py::ValueError{ "`prior` must be a dict of {topic_id: float}" };
			priors[k] = v;
		}
		inst->setWordPrior(word, priors);
	}
	else
	{
		vector<tomoto::Float> priorVal;
		if (!py::toCpp(prior, priorVal))
		{
			throw py::ValueError{ "`prior` must be a list of floats with len = k" };
		}
		inst->setWordPrior(word, priorVal);
	}
}

std::vector<tomoto::Float> LDAModelObject::getWordPrior(const std::string& word) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	return inst->getWordPrior(word);
}

void LDAModelObject::train(size_t iteration, size_t workers, size_t ps, bool freezeTopics, size_t callbackInterval, PyObject* callback)
{
	if (seedGiven && workers != 1 && PyErr_WarnEx(PyExc_RuntimeWarning, "The training result may differ even with fixed seed if `workers` != 1.", 1)) throw py::ExcPropagation{};
	auto* inst = getInst<tomoto::ILDAModel>();
	if (!isPrepared)
	{
		inst->prepare(true, minWordCnt, minWordDf, removeTopWord);
		isPrepared = true;
	}

	if (callback == Py_None) callback = nullptr;
	if (callback && !PyCallable_Check(callback)) throw py::ValueError{ "`callback` should be a callable object" };
	if (!callback || callbackInterval <= 0)
	{
		callbackInterval = iteration;
	}

	for (size_t it = 0; it < iteration; it += callbackInterval)
	{
		if (callback)
		{
			py::UniqueObj args{ py::buildPyTuple(getObject(), it, iteration)};
			if (callback)
			{
				py::UniqueObj ret{ PyObject_CallObject(callback, args.get()) };
				if (!ret) throw py::ExcPropagation{};
			}
		}

		if (inst->train(std::min(callbackInterval, iteration - it), workers, (tomoto::ParallelScheme)ps, !!fixed) < 0)
		{
			throw py::RuntimeError{ "Train failed" };
		}
	}
	if (callback)
	{
		py::UniqueObj args{ py::buildPyTuple(getObject(), iteration, iteration) };
		if (callback)
		{
			py::UniqueObj ret{ PyObject_CallObject(callback, args.get()) };
			if (!ret) throw py::ExcPropagation{};
		}
	}
}

py::UniqueObj LDAModelObject::getTopicWords(size_t topicId, size_t topN, bool returnId) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "topic_id must < K" };
		
	if (returnId)
	{
		return py::buildPyValue(inst->getWordIdsByTopicSorted(topicId, topN));
	}
	else
	{
		return py::buildPyValue(inst->getWordsByTopicSorted(topicId, topN));
	}
}

std::vector<tomoto::Float> LDAModelObject::getTopicWordDist(size_t topicId, bool normalize) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "topic_id must < K" };

	return inst->getWidsByTopic(topicId, normalize);
}

py::UniqueObj LDAModelObject::infer(PyObject* docObj, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (!isPrepared) throw py::RuntimeError{ "cannot infer with untrained model" };
	py::UniqueObj iter;
	if (PyObject_TypeCheck(docObj, py::Type<CorpusObject>))
	{
		auto cps = makeCorpus(docObj, transform);
		std::vector<tomoto::DocumentBase*> docs;
		for (auto& d : cps->docsMade) docs.emplace_back(d.get());
		auto ll = inst->infer(docs, iteration, tolerance, workers, ps, together);
		return py::buildPyTuple(cps, ll);
	}
	else if (auto* doc = py::checkType<DocumentObject>(docObj))
	{
		if (doc->corpus->tm.get() != py::getPObjectAddress(this)) throw py::ValueError{ "`doc` was from another model, not fit to this model" };
		if (doc->owner)
		{
			std::vector<tomoto::DocumentBase*> docs;
			docs.emplace_back((tomoto::DocumentBase*)doc->getBoundDoc());
			float ll = inst->infer(docs, iteration, tolerance, workers, ps, together)[0];
			doc->initialized = true;
			return py::buildPyTuple(py::buildPyValue(inst->getTopicsByDoc(doc->getBoundDoc())), ll);
		}
		else
		{
			return py::buildPyTuple(py::buildPyValue(inst->getTopicsByDoc(doc->getBoundDoc())), nullptr);
		}
	}
	else if (py::clearError(), (iter = py::UniqueObj{ PyObject_GetIter(docObj) }))
	{
		std::vector<tomoto::DocumentBase*> docs;
		std::vector<DocumentObject*> docObjs;
		py::UniqueObj item;
		while ((item = py::UniqueObj{ PyIter_Next(iter.get()) }))
		{
			auto* doc = py::checkType<DocumentObject>(item.get());
			if (!doc) throw py::ValueError{ "`doc` must be tomotopy.Document type or list of tomotopy.Document" };
			if (doc->corpus->tm.get() != py::getPObjectAddress(this)) throw py::ValueError{ "`doc` was from another model, not fit to this model" };
			docs.emplace_back((tomoto::DocumentBase*)doc->doc);
			docObjs.emplace_back(doc);
		}
		if (PyErr_Occurred()) throw py::ExcPropagation{};
		auto ll = inst->infer(docs, iteration, tolerance, workers, ps, together);
			
		for (auto doc : docObjs) doc->initialized = true;

		auto ret = py::UniqueObj{ PyList_New(docs.size()) };
		size_t i = 0;
		for (auto d : docs)
		{
			PyList_SetItem(ret.get(), i++, py::buildPyValue(inst->getTopicsByDoc(d)).release());
		}
		if (together)
		{
			return py::buildPyTuple(ret, ll[0]);
		}
		else
		{
			return py::buildPyTuple(ret, ll);
		}
	}
	else
	{
		throw py::ValueError{ "`doc` must be tomotopy.Document type or list of tomotopy.Document" };
	}
}

void LDAModelObject::save(const std::string& filename, const std::vector<uint8_t>& extraData, bool full)
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (!isPrepared)
	{
		inst->prepare(true, minWordCnt, minWordDf, removeTopWord);
		isPrepared = true;
	}
	ofstream str{ filename, ios_base::binary };
	if (!str) throw py::OSError{ std::string("cannot open file '") + filename + std::string("'") };

	inst->saveModel(str, !!full, &extraData);
}

py::UniqueObj LDAModelObject::saves(const std::vector<uint8_t>& extraData, bool full)
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (!isPrepared)
	{
		inst->prepare(true, minWordCnt, minWordDf, removeTopWord);
		isPrepared = true;
	}
	ostringstream str;
	inst->saveModel(str, !!full, &extraData);
	return py::UniqueObj{ PyBytes_FromStringAndSize(str.str().data(), str.str().size()) };
}

void LDAModelObject::updateVocab(const std::vector<std::string>& newVocabs)
{
	auto* inst = getInst<tomoto::ILDAModel>();
	inst->updateVocab(newVocabs);
}

py::UniqueCObj<CorpusObject> LDAModelObject::getDocs() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	auto ret = py::UniqueCObj<CorpusObject>{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr) };
	return ret;
}

py::UniqueCObj<VocabObject> LDAModelObject::getVocabs() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	auto ret = py::makeNewObject<VocabObject>();
	ret->dep = py::UniqueObj{ getObject() };
	Py_INCREF(ret->dep);
	ret->vocabs = (tomoto::Dictionary*)&inst->getVocabDict();
	ret->size = inst->getVocabDict().size();
	return ret;
}

py::UniqueCObj<VocabObject> LDAModelObject::getUsedVocabs() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	auto ret = py::makeNewObject<VocabObject>();
	ret->dep = py::UniqueObj{ getObject() };
	Py_INCREF(ret->dep);
	ret->vocabs = (tomoto::Dictionary*)&inst->getVocabDict();
	ret->size = inst->getV();
	return ret;
}

py::UniqueObj LDAModelObject::getUsedVocabCf() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	return py::buildPyValue(inst->getVocabCf().begin(), inst->getVocabCf().begin() + inst->getV());
}

std::vector<double> LDAModelObject::getUsedVocabWeightedCf() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	return inst->getVocabWeightedCf();
}

py::UniqueObj LDAModelObject::getUsedVocabDf() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	return py::buildPyValue(inst->getVocabDf().begin(), inst->getVocabDf().begin() + inst->getV());
}

std::vector<uint64_t> LDAModelObject::getCountByTopics() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	return inst->getCountByTopic();
}

std::vector<float> LDAModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	vector<float> ret;
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		ret.emplace_back(inst->getAlpha(i));
	}
	return ret;
}

std::vector<std::string> LDAModelObject::getRemovedTopWords() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	vector<string> ret;
	size_t last = inst->getVocabDict().size();
	for (size_t rmV = last - removeTopWord; rmV < last; ++rmV)
	{
		ret.emplace_back(inst->getVocabDict().toWord(rmV));
	}
	return ret;
}

py::UniqueObj LDAModelObject::getWordForms(size_t idx) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	if (idx == (size_t)-1)
	{
		return py::buildPyValue(inst->getWordFormCnts());
	}
	else
	{
		if (idx >= inst->getWordFormCnts().size())
		{
			throw py::ValueError{ "`idx` must be less than the `len(used_vocabs)`." };
		}
		return py::buildPyValue(inst->getWordFormCnts()[idx]);
	}
}

py::UniqueObj LDAModelObject::getHash() const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	std::array<uint64_t, 2> hash = inst->getHash();
	return py::UniqueObj{ PyObject_CallMethod((PyObject*)&PyLong_Type, "from_bytes", "y#s", (const char*)hash.data(), sizeof(hash), "big") };
}

py::UniquePObj<LDAModelObject> LDAModelObject::copy(PyObject* cls) const
{
	auto* inst = getInst<tomoto::ILDAModel>();
	auto ret = py::makeNewObject<py::PObject<LDAModelObject>>((PyTypeObject*)cls);
	ret->inst = inst->copy();
	ret->isPrepared = isPrepared;
	ret->minWordCnt = minWordCnt;
	ret->minWordDf = minWordDf;
	ret->removeTopWord = removeTopWord;
	return ret;
}

std::pair<py::UniquePObj<LDAModelObject>, std::vector<uint8_t>> LDAModelObject::load(PyObject* cls, const std::string& filename)
{
	ifstream str{ filename, ios_base::binary };
	if (!str) throw ios_base::failure{ std::string("cannot open file '") + filename + std::string("'") };
	for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)
	{
		str.seekg(0);
		py::UniqueObj args{ py::buildPyTuple(i) };
		py::UniqueObj newInst{ PyObject_CallObject(cls, args.get()) };
		if (!newInst) throw py::ExcPropagation{};
		auto p = py::checkType<py::PObject<LDAModelObject>>(std::move(newInst));
		auto inst = p->getInst<tomoto::ILDAModel>();
		vector<uint8_t> extraData;
		try
		{
			extraData.clear();
			inst->loadModel(str, &extraData);
		}
		catch (const tomoto::serializer::UnfitException&)
		{
			continue;
		}
		p->isPrepared = true;
		return make_pair(std::move(p), std::move(extraData));
	}
	throw runtime_error{ std::string("'") + filename + std::string("' is not valid model file") };
}

std::pair<py::UniquePObj<LDAModelObject>, std::vector<uint8_t>> LDAModelObject::loads(PyObject* cls, const std::vector<uint8_t>& data)
{
	tomoto::serializer::imstream str{ (const char*)data.data(), (std::ptrdiff_t)data.size() };
	for (size_t i = 0; i < (size_t)tomoto::TermWeight::size; ++i)
	{
		str.seekg(0);
		py::UniqueObj args{ py::buildPyTuple(i) };
		py::UniqueObj newInst{ PyObject_CallObject(cls, args.get()) };
		if (!newInst) throw py::ExcPropagation{};
		auto p = py::checkType<py::PObject<LDAModelObject>>(std::move(newInst));
		auto inst = p->getInst<tomoto::ILDAModel>();
		vector<uint8_t> extraData;
		try
		{
			extraData.clear();
			inst->loadModel(str, &extraData);
		}
		catch (const tomoto::serializer::UnfitException&)
		{
			continue;
		}
		p->isPrepared = true;
		return make_pair(std::move(p), std::move(extraData));
	}
	throw runtime_error{ "`data` is not valid model file" };
}

py::UniqueObj DocumentObject::getZFromLDA() const
{
	return docVisit<tomoto::DocumentLDA>(getBoundDoc(), [](auto* doc)
	{
		return buildPyValueReorder(doc->Zs, doc->wOrder, [](tomoto::Tid x) -> int16_t { return x; });
	});
}

std::vector<float> DocumentObject::getCountVector() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "This method can only be called by documents bound to the topic model." };
	if (!corpus->tm->inst) throw runtime_error{ "inst is null" };
	size_t v = corpus->tm->inst->getV();

	return docVisit<tomoto::DocumentLDA>(getBoundDoc(), [&](auto* doc)
	{
		return doc->getCountVector(v);
	});

	throw py::AttributeError{ "cannot get count vector" };
}
