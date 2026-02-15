#include "../../TopicModel/HLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

HLDAModelObject::HLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t depth, PyObject* alpha, float eta, float gamma,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::HLDAArgs margs;
	margs.k = depth;
	if (alpha)
	{
		margs.alpha = broadcastObj<tomoto::Float>(alpha, margs.k,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `depth` (given " + py::repr(alpha) + ")"; }
		);
	}
	margs.eta = eta;
	margs.gamma = gamma;
	if (seed && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw invalid_argument{ "`seed` must be an integer or None." };
	}

	inst = tomoto::IHLDAModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

py::UniqueObj DocumentObject::getZFromHLDA() const
{
	return docVisit<tomoto::DocumentHLDA>(getBoundDoc(), [](auto* doc)
	{
		return buildPyValueReorder(doc->Zs, doc->wOrder, [doc](tomoto::Tid x) -> int16_t
		{ 
			if (x == tomoto::non_topic_id) return -1;
			return doc->path[x]; 
		});
	});
}


std::vector<float> HLDAModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::IHLDAModel>();
	vector<float> ret;
	for (size_t i = 0; i < inst->getLevelDepth(); ++i)
	{
		ret.emplace_back(inst->getAlpha(i));
	}
	return ret;
}

std::vector<int32_t> DocumentObject::getPath() const 
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `path` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentHLDA>(getBoundDoc(), [](auto* doc) -> std::optional<std::vector<int32_t>>
	{ 
		return doc->path;
	})) return *ret; 
	throw py::AttributeError{ "doc has no `path` field!" };
}
