#include "../../TopicModel/CT.h"

#include "module.h"
#include "utils.h"

using namespace std;

CTModelObject::CTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, PyObject* alpha, float eta,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::CTArgs margs;
	margs.k = k;
	if (alpha) margs.alpha = broadcastObj<tomoto::Float>(alpha, margs.k,
		[=]() { return "`smoothing_alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
	);
	margs.eta = eta;
	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	inst = tomoto::ICTModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

py::UniqueObj CTModelObject::getCorrelations(PyObject* topicId) const
{
	auto* inst = getInst<tomoto::ICTModel>();
	if (!topicId || topicId == Py_None)
	{
		float* ptr;
		auto ret = py::newEmptyArray(ptr, 2, inst->getK(), inst->getK());
		for (size_t i = 0; i < inst->getK(); ++i)
		{
			auto l = inst->getCorrelationTopic(i);
			memcpy(ptr + i * inst->getK(), l.data(), sizeof(float) * inst->getK());
		}
		return ret;
	}

	size_t topicIdVal;
	if (!py::toCpp(topicId, topicIdVal))
	{
		throw py::ValueError{ "`topic_id` must be an integer or None." };
	}
	if (topicIdVal >= inst->getK()) throw py::ValueError{ "`topic_id` must be in range [0, `k`)" };
	return py::buildPyValue(inst->getCorrelationTopic(topicIdVal));
}

py::UniqueObj CTModelObject::getPriorCov() const
{
	auto* inst = getInst<tomoto::ICTModel>();
	float* ptr;
	auto ret = py::newEmptyArray(ptr, inst->getK(), inst->getK());
	auto cov = inst->getPriorCov();
	memcpy(ptr, cov.data(), sizeof(float) * inst->getK() * inst->getK());
	return ret;
}

py::UniqueObj DocumentObject::getBeta() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `beta` field!" };
	if (!doc) throw py::RuntimeError{ "doc is null!" };

	if (auto ret = docVisit<tomoto::DocumentCTM>(getBoundDoc(), [](auto* doc)
	{
		return py::buildPyValueTransform(
			doc->smBeta.data(), doc->smBeta.data() + doc->smBeta.size(),
			logf
		);
	})) return ret;
	throw py::AttributeError{ "doc has no `beta` field!" };
}
