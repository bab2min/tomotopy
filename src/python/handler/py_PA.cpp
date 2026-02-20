#include "../../TopicModel/PA.h"

#include "module.h"
#include "utils.h"

using namespace std;

PAModelObject::PAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k1, size_t k2,
	PyObject* alpha, PyObject* subAlpha, float eta,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::PAArgs margs;
	margs.k = k1;
	margs.k2 = k2;
	if (alpha)
	{
		margs.alpha = broadcastObj<tomoto::Float>(alpha, k1,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k1` (given " + py::repr(alpha) + ")"; }
		);
	}
	if (subAlpha)
	{
		margs.subalpha = broadcastObj<tomoto::Float>(subAlpha, k2,
			[=]() { return "`subalpha` must be an instance of `float` or `List[float]` with length `k2` (given " + py::repr(subAlpha) + ")"; }
		);
	}
	margs.eta = eta;

	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	auto inst = tomoto::IPAModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	this->inst = move(inst);
	this->isPrepared = false;
	this->seedGiven = !!seed;
	this->minWordCnt = minCnt;
	this->minWordDf = minDf;
	this->removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

std::vector<float> PAModelObject::getSubTopicPrior(size_t topicId, bool normalize) const
{
	auto* inst = getInst<tomoto::IPAModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < k1" };

	return inst->getSubTopicBySuperTopic(topicId, normalize);
}

std::vector<std::pair<tomoto::Tid, tomoto::Float>> PAModelObject::getSubTopics(size_t topicId, size_t topN) const
{
	auto* inst = getInst<tomoto::IPAModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < k1" };

	return inst->getSubTopicBySuperTopicSorted(topicId, topN);
}

std::vector<std::pair<std::string, tomoto::Float>> PAModelObject::getTopicWords(size_t topicId, size_t topN) const
{
	auto* inst = getInst<tomoto::IPAModel>();
	if (topicId >= inst->getK2()) throw py::ValueError{ "must topic_id < k2" };

	return inst->getWordsByTopicSorted(topicId, topN);
}

std::vector<float> PAModelObject::getTopicWordDist(size_t topicId, bool normalize) const
{
	auto* inst = getInst<tomoto::IPAModel>();
	if (topicId >= inst->getK2()) throw py::ValueError{ "must topic_id < k2" };
	return inst->getWidsByTopic(topicId, normalize);
}

std::vector<std::pair<tomoto::Tid, tomoto::Float>> DocumentObject::getSubTopics(size_t topN) const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "This method can only be called by documents bound to the topic model." };
	auto* inst = corpus->tm->getInst<tomoto::IPAModel>();
	if (!corpus->tm->isPrepared) throw py::RuntimeError{ "train() should be called first for calculating the topic distribution" };
	return inst->getSubTopicsByDocSorted(getBoundDoc(), topN);
}

std::vector<float> DocumentObject::getSubTopicDist(bool normalize) const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "This method can only be called by documents bound to the topic model." };
	auto* inst = corpus->tm->getInst<tomoto::IPAModel>();
	if (!corpus->tm->isPrepared) throw py::RuntimeError{ "train() should be called first for calculating the topic distribution" };
	return inst->getSubTopicsByDoc(getBoundDoc(), normalize);
}

py::UniqueObj PAModelObject::infer(PyObject* docObj, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const
{
	auto* inst = getInst<tomoto::IPAModel>();
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
		if (doc->corpus->tm.get() != py::getPObjectAddress<LDAModelObject>(this)) throw py::ValueError{ "`doc` was from another model, not fit to this model" };
		if (doc->owner)
		{
			std::vector<tomoto::DocumentBase*> docs;
			docs.emplace_back((tomoto::DocumentBase*)doc->getBoundDoc());
			float ll = inst->infer(docs, iteration, tolerance, workers, ps, together)[0];
			doc->initialized = true;
			return py::buildPyTuple(py::buildPyTuple(
				inst->getTopicsByDoc(doc->getBoundDoc()),
				inst->getSubTopicsByDoc(doc->getBoundDoc())
			), ll);
		}
		else
		{
			return py::buildPyTuple(py::buildPyTuple(
				inst->getTopicsByDoc(doc->getBoundDoc()),
				inst->getSubTopicsByDoc(doc->getBoundDoc())
			), nullptr);
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
			if (doc->corpus->tm.get() != py::getPObjectAddress<LDAModelObject>(this)) throw py::ValueError{ "`doc` was from another model, not fit to this model" };
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
			PyList_SetItem(ret.get(), i++, py::buildPyTuple(
				inst->getTopicsByDoc(d),
				inst->getSubTopicsByDoc(d)
			).release());
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

std::vector<uint64_t> PAModelObject::getCountBySuperTopic() const
{
	auto* inst = getInst<tomoto::IPAModel>();
	return inst->getCountBySuperTopic();
}

py::UniqueObj PAModelObject::getSubAlpha() const
{
	auto* inst = getInst<tomoto::IPAModel>();
	float* ptr;
	auto ret = py::newEmptyArray<float>(ptr, inst->getK(), inst->getK2());
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		auto l = inst->getSubAlpha(i);
		memcpy(ptr + i * inst->getK2(), l.data(), sizeof(float) * inst->getK2());
	}
	return ret;
}

py::UniqueObj DocumentObject::getZ2() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `Z2s` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentPA>(getBoundDoc(), [](auto* doc) 
	{ 
		return buildPyValueReorder(doc->Z2s, doc->wOrder); 
	})) return ret; 
	throw py::AttributeError{ "doc has no `Z2s` field!" };
}