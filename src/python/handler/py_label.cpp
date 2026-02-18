#include "module.h"
#include "label.h"
#include "utils.h"
#include "label_docs.h"

using namespace std;

const string& CandWordIterator::operator*() const
{
	auto& v = co->tm ? co->tm->inst->getVocabDict() : *co->corpus->vocab->vocabs;
	return v.toWord(co->cand.w[idx]);
}

std::string CandidateObject::repr() const
{
	string ret = "tomotopy.label.Candidate(words=[";
	for (auto& w : *this)
	{
		ret.push_back('"');
		ret += w;
		ret.push_back('"');
		ret.push_back(',');
	}
	ret.back() = ']';
	ret += ", name=\"";
	ret += cand.name;
	ret += "\", score=";
	ret += to_string(cand.score);
	ret.push_back(')');
	return ret;
}

py::UniqueObj CandidateObject::getWords() const
{
	return py::buildPyValue(begin(), end());
}

std::string CandidateObject::getName() const
{
	return cand.name;
}

void CandidateObject::setName(const std::string& name)
{
	cand.name = name;
}

float CandidateObject::getScore() const
{
	return cand.score;
}

size_t CandidateObject::getCf() const
{
	return cand.cf;
}

size_t CandidateObject::getDf() const
{
	return cand.df;
}

struct PMIExtractorObject : public py::CObject<PMIExtractorObject>
{
	std::unique_ptr<tomoto::label::IExtractor> inst;

	PMIExtractorObject() = default;

	using _InitArgs = std::tuple<size_t, size_t, size_t, size_t, size_t, bool>;
	PMIExtractorObject(
		size_t minCf, size_t minDf, size_t minLen, size_t maxLen, size_t maxCand, bool normalized
	)
		: inst{ std::make_unique<tomoto::label::PMIExtractor>(minCf, minDf, minLen, maxLen, maxCand, normalized) }
	{
	}

	py::UniqueObj extract(PyObject* arg)
	{
		auto tm = py::checkType<py::PObject<LDAModelObject>>(arg, "tm must be a tomotopy.LDAModel instance.");
		auto cands = inst->extract(tm->value.inst.get());
		py::UniqueObj ret = py::UniqueObj{ PyList_New(0) };
		for (auto& c : cands)
		{
			auto item = py::makeNewObject<CandidateObject>();
			item->tm = py::UniquePObj<LDAModelObject>{ tm };
			item->tm.incref();
			item->cand = move(c);
			PyList_Append(ret.get(), (PyObject*)item.get());
		}
		return ret;
	}
};

struct FoRelevanceObject : public py::CObject<FoRelevanceObject>
{
	std::unique_ptr<tomoto::label::ILabeler> inst;
	py::UniquePObj<LDAModelObject> tm;

	FoRelevanceObject() = default;

	using _InitArgs = std::tuple<PyObject*, PyObject*, size_t, float, float, size_t, size_t>;
	FoRelevanceObject(
		PyObject* topicModel, PyObject* cands, size_t minDf, float smoothing, float mu, size_t windowSize, size_t numWorkers
	)
	{
		tm = py::checkType<py::PObject<LDAModelObject>>(py::UniqueObj{ topicModel }, "topicModel must be a tomotopy.LDAModel instance.");
		tm.incref();
		py::UniqueObj iter{ PyObject_GetIter(cands) };
		if (!iter) throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
		vector<tomoto::label::Candidate*> pcands;
		{
			py::UniqueObj item;
			while ((item = py::UniqueObj{ PyIter_Next(iter.get()) }))
			{
				if (!PyObject_TypeCheck(item, py::Type<CandidateObject>))
				{
					throw py::ValueError{ "`cands` must be an iterable of `tomotopy.label.Candidate`" };
				}
				pcands.emplace_back(&((CandidateObject*)item.get())->cand);
			}
		}
		auto deref = [](tomoto::label::Candidate* p)->tomoto::label::Candidate& { return *p; };
		inst = make_unique<tomoto::label::FoRelevance>(
			tm->inst.get(),
			tomoto::makeTransformIter(pcands.begin(), deref),
			tomoto::makeTransformIter(pcands.end(), deref),
			minDf, smoothing, 0, mu, windowSize, numWorkers
		);
	}

	std::vector<std::pair<std::string, tomoto::Float>> getTopicLabels(size_t k, size_t topN = 10) const
	{
		return inst->getLabels(k, topN);
	}
};

void addLabelTypes(py::Module& module)
{
	module.addType(
		py::define<CandidateObject>("tomotopy.label.Candidate", "_LabelCandidate", Py_TPFLAGS_BASETYPE)
			.property<&CandidateObject::getWords>("words")
			.property<&CandidateObject::getName, &CandidateObject::setName>("name")
			.property<&CandidateObject::getScore>("score")
			.property<&CandidateObject::getCf>("cf")
			.property<&CandidateObject::getDf>("df")	
	);

	module.addType(
		py::define<PMIExtractorObject>("tomotopy.label.PMIExtractor", "_LabelPMIExtractor", Py_TPFLAGS_BASETYPE)
			.method<&PMIExtractorObject::extract>("extract")
	);

	module.addType(
		py::define<FoRelevanceObject>("tomotopy.label.FoRelevance", "_LabelFoRelevance", Py_TPFLAGS_BASETYPE)
			.method<&FoRelevanceObject::getTopicLabels>("get_topic_labels")
	);
}
