#include "../../TopicModel/HPA.h"

#include "module.h"
#include "utils.h"

using namespace std;

HPAModelObject::HPAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k1, size_t k2,
	PyObject* alpha, PyObject* subAlpha, float eta,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::HPAArgs margs;
	margs.k = k1;
	margs.k2 = k2;
	if (alpha)
	{
		margs.alpha = broadcastObj<tomoto::Float>(alpha, k1 + 1,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k1 + 1` (given " + py::repr(alpha) + ")"; }
		);
	}
	if (subAlpha)
	{
		margs.subalpha = broadcastObj<tomoto::Float>(subAlpha, k2 + 1,
			[=]() { return "`subalpha` must be an instance of `float` or `List[float]` with length `k2 + 1` (given " + py::repr(subAlpha) + ")"; }
		);
	}
	margs.eta = eta;

	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	auto inst = tomoto::IHPAModel::create((tomoto::TermWeight)tw, false, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	this->inst = move(inst);
	this->isPrepared = false;
	this->seedGiven = !!seed;
	this->minWordCnt = minCnt;
	this->minWordDf = minDf;
	this->removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

std::vector<std::pair<std::string, tomoto::Float>> HPAModelObject::getTopicWords(size_t topicId, size_t topN) const
{
	auto* inst = getInst<tomoto::IHPAModel>();
	if (topicId > inst->getK() + inst->getK2()) throw py::ValueError{ "must topic_id < 1 + K1 + K2" };
	return inst->getWordsByTopicSorted(topicId, topN);
}

std::vector<float> HPAModelObject::getTopicWordDist(size_t topicId, bool normalize) const
{
	auto* inst = getInst<tomoto::IHPAModel>();
	if (topicId > inst->getK() + inst->getK2()) throw py::ValueError{ "must topic_id < 1 + K1 + K2" };
	return inst->getWidsByTopic(topicId, normalize);
}

py::UniqueObj HPAModelObject::infer(PyObject* doc, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const
{
	return LDAModelObject::infer(doc, iteration, tolerance, workers, ps, together, transform);
}

py::UniqueObj HPAModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::IHPAModel>();
	float* ptr;
	auto ret = py::newEmptyArray<float>(ptr, inst->getK() + 1);
	for (size_t i = 0; i <= inst->getK(); ++i)
	{
		ptr[i] = inst->getAlpha(i);
	}
	return ret;
}

py::UniqueObj HPAModelObject::getSubAlpha() const
{
	auto* inst = getInst<tomoto::IHPAModel>();
	float* ptr;
	auto ret = py::newEmptyArray<float>(ptr, inst->getK(), inst->getK2() + 1);
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		auto l = inst->getSubAlpha(i);
		memcpy(ptr + i * (inst->getK2() + 1), l.data(), sizeof(float) * (inst->getK2() + 1));
	}
	return ret;
}
