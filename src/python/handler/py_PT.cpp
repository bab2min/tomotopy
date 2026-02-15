#include "../../TopicModel/PT.h"

#include "module.h"
#include "utils.h"

using namespace std;

PTModelObject::PTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, size_t p, PyObject* alpha, float eta,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::PTArgs margs;
	margs.k = k;
	if (alpha)
	{
		margs.alpha = broadcastObj<tomoto::Float>(alpha, margs.k,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
		);
	}
	margs.eta = eta;
	if (seed && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw invalid_argument{ "`seed` must be an integer or None." };
	}

	if (margs.p == 0) margs.p = margs.k * 10;

	inst = tomoto::IPTModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

size_t DocumentObject::getPseudoDocId() const 
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `pseudoDoc` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentPT>(getBoundDoc(), [](auto* doc) -> std::optional<size_t>
	{ 
		return doc->pseudoDoc;
	})) return *ret; 
	throw py::AttributeError{ "doc has no `pseudoDoc` field!" };
}

std::vector<std::pair<tomoto::Tid, tomoto::Float>> DocumentObject::getTopicsFromPseudoDoc(size_t topN) const
{
	auto* mdl = corpus->tm->getInstDynamic<tomoto::IPTModel>();
	if (!mdl) throw py::ValueError{ "`from_pseudo_doc` is valid for only `tomotopy.PTModel`." };
	return mdl->getTopicsByDocSorted(getBoundDoc(), topN);
}

std::vector<float> DocumentObject::getTopicDistFromPseudoDoc(bool normalize) const
{
	auto* mdl = corpus->tm->getInstDynamic<tomoto::IPTModel>();
	if (!mdl) throw py::ValueError{ "`from_pseudo_doc` is valid for only `tomotopy.PTModel`." };
	return mdl->getTopicsByDoc(getBoundDoc(), normalize);
}
