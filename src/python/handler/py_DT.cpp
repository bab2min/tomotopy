#include "../../TopicModel/DT.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType DTModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["timepoint"] = getValueFromMiscDefault<uint32_t>("timepoint", o, "`DTModel` requires a `timepoint` value in `int` type.");
	return ret;
}

DTModelObject::DTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, size_t t, float alphaVar, float etaVar, float phiVar,
	float lrA, float lrB, float lrC,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::DTArgs margs;
	margs.k = k;
	margs.t = t;
	margs.alpha[0] = alphaVar;
	margs.eta = etaVar;
	margs.phi = phiVar;
	margs.shapeA = lrA;
	margs.shapeB = lrB;
	margs.shapeC = lrC;

	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	inst = tomoto::IDTModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::RuntimeError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;
	insertCorpus(corpus, transform);
}

std::optional<size_t> DTModelObject::addDoc(PyObject* words, size_t timepoint, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::IDTModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["timepoint"] = (uint32_t)timepoint;
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

py::UniqueCObj<DocumentObject> DTModelObject::makeDoc(PyObject* words, size_t timepoint)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::IDTModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["timepoint"] = (uint32_t)timepoint;
	auto doc = inst->makeDoc(raw);
	py::UniqueCObj<CorpusObject> corpus{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr) };
	auto ret = py::makeNewObject<DocumentObject>(getDocumentCls());
	ret->corpus = corpus.copy();
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

std::vector<float> DTModelObject::getAlpha2(size_t timepoint) const
{
	auto* inst = getInst<tomoto::IDTModel>();
	if (timepoint >= inst->getT()) throw py::ValueError{ "`timepoint` must < `DTModel.num_timepoints`" };

	vector<float> alphas;
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		alphas.emplace_back(inst->getAlpha(i, timepoint));
	}
	return alphas;
}

std::vector<float> DTModelObject::getPhi(size_t timepoint, size_t topicId) const
{
	auto* inst = getInst<tomoto::IDTModel>();
	return inst->getPhi(topicId, timepoint);
}

std::vector<std::pair<std::string, float>> DTModelObject::getTopicWords(size_t topicId, size_t timepoint, size_t topN) const
{
	auto* inst = getInst<tomoto::IDTModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < k" };
	if (timepoint >= inst->getT()) throw py::ValueError{ "must topic_id < t" };
	return inst->getWordsByTopicSorted(topicId + inst->getK() * timepoint, topN);
}

std::vector<float> DTModelObject::getTopicWordDist(size_t topicId, size_t timepoint, bool normalize) const
{
	auto* inst = getInst<tomoto::IDTModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < k" };
	if (timepoint >= inst->getT()) throw py::ValueError{ "must topic_id < t" };
	return inst->getWidsByTopic(topicId + inst->getK() * timepoint, normalize);
}

py::UniqueObj DTModelObject::getCountByTopic() const
{
	auto* inst = getInst<tomoto::IDTModel>();
	auto l = inst->getCountByTopic();

	int64_t* ptr;
	auto ret = py::newEmptyArray(ptr, inst->getT(), inst->getK());
	for (size_t i = 0; i < inst->getT(); ++i)
	{
		memcpy(ptr + i * inst->getK(), &l[inst->getK() * i], sizeof(int64_t) * inst->getK());
	}
	return ret;
}

py::UniqueObj DTModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::IDTModel>();
	float* ptr;
	auto ret = py::newEmptyArray(ptr, inst->getT(), inst->getK());
	for (size_t t = 0; t < inst->getT(); ++t)
	{
		for (size_t k = 0; k < inst->getK(); ++k)
		{
			ptr[t * inst->getK() + k] = inst->getAlpha(k, t);
		}
	}
	return ret;
}

py::UniqueObj DocumentObject::getEta() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `eta` field!" };
	if (!doc) throw py::RuntimeError{ "doc is null!" };

	if (auto ret = docVisit<tomoto::DocumentDTM>(getBoundDoc(), [](auto* doc)
	{
		return py::buildPyValue(doc->eta.array().data(), doc->eta.array().data() + doc->eta.array().size());
	})) return ret;

	throw py::AttributeError{ "doc has no `eta` field!" };
}

size_t DocumentObject::getTimepoint() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `timepoint` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentDTM>(getBoundDoc(), [](auto* doc) -> std::optional<size_t>
	{
		return doc->timepoint; 
	})) return *ret; 
	throw py::AttributeError{ "doc has no `timepoint` field!" };
}
