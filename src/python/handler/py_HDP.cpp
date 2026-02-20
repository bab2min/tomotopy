#include "../../TopicModel/HDP.h"

#include "module.h"
#include "utils.h"

using namespace std;

HDPModelObject::HDPModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t initialK, float alpha, float eta, float gamma,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::HDPArgs margs;
	margs.k = initialK;
	margs.alpha[0] = alpha;
	margs.eta = eta;
	margs.gamma = gamma;
	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	inst = tomoto::IHDPModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;
	insertCorpus(corpus, transform);
}

bool HDPModelObject::isLiveTopic(size_t topicId) const
{
	auto* inst = getInst<tomoto::IHDPModel>();
	if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < K" };
	return inst->isLiveTopic(topicId);
}

std::pair<py::UniquePObj<LDAModelObject>, py::UniqueObj> HDPModelObject::convertToLDA(PyObject* LDAType, float topicThreshold) const
{
	auto* inst = getInst<tomoto::IHDPModel>();
	std::vector<tomoto::Tid> newK;
	auto lda = inst->convertToLDA(topicThreshold, newK);
	auto ret = py::makeNewObject<py::PObject<LDAModelObject>>((PyTypeObject*)LDAType);
	ret->inst = move(lda);
	ret->isPrepared = true;
	ret->minWordCnt = minWordCnt;
	ret->minWordDf = minWordDf;
	ret->removeTopWord = removeTopWord;
	return make_pair(move(ret), py::buildPyValue(newK, py::cast_to_signed));
}

std::vector<int32_t> HDPModelObject::purgeDeadTopics()
{
	auto* inst = getInst<tomoto::IHDPModel>();
	std::vector<int32_t> ret;
	for (auto t : inst->purgeDeadTopics())
	{
		ret.emplace_back((int16_t)t);
	}
	return ret;
}

float HDPModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::IHDPModel>();
	return inst->getAlpha();
}

py::UniqueObj DocumentObject::getZFromHDP() const
{
	return docVisit<tomoto::DocumentHDP>(getBoundDoc(), [](auto* doc)
	{
		return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](tomoto::Tid x) -> int16_t
		{ 
			if (x == tomoto::non_topic_id) return -1;
			return doc->numTopicByTable[x].topic; 
		});
	});
}
