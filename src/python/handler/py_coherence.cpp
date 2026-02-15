
#include "module.h"
#include "coherence.h"

using namespace std;

CoherenceObject::CoherenceObject(PyObject* corpus,
	ProbEstimation pe, Segmentation seg, ConfirmMeasure cm, IndirectMeasure im,
	size_t windowSize, double eps, double gamma, PyObject* targets)
	: model{ pe, windowSize }
{
	if (!PyObject_TypeCheck(corpus, py::Type<CorpusObject>))
	{
		throw py::ValueError{ "`corpus` must be an instance of `tomotopy.utils.Corpus`." };
	}

	this->corpus = py::UniqueCObj<CorpusObject>{ (CorpusObject*)corpus };
	Py_INCREF(corpus);

	vector<tomoto::Vid> targetIds;
	py::foreach<string>(targets, [&](const string& w)
	{
		auto wid = this->corpus->getVocabDict().toWid(w);
		if (wid != tomoto::non_vocab_id) targetIds.emplace_back(wid);
	}, "`targets` must be an iterable of `str`.");

	this->model.insertTargets(targetIds.begin(), targetIds.end());

	for (size_t i = 0; i < this->corpus->len(); ++i)
	{
		auto* doc = this->corpus->getDoc(i);
		this->model.insertDoc(
			wordBegin(doc, this->corpus->isIndependent()),
			wordEnd(doc, this->corpus->isIndependent())
		);
	}

	this->seg = seg;
	this->cm = tomoto::coherence::AnyConfirmMeasurer::getInstance(cm, im, targetIds.begin(), targetIds.end(), eps, gamma);
}

double CoherenceObject::getScore(PyObject* words) const
{
	vector<tomoto::Vid> wordIds;
	py::foreach<string>(words, [&](const string& w)
	{
		auto wid = corpus->getVocabDict().toWid(w);
		if (wid != tomoto::non_vocab_id) wordIds.emplace_back(wid);
	}, "`words` must be an iterable of `str`.");

	switch (seg)
	{
	case Segmentation::one_one:
		return model.template getScore<Segmentation::one_one>(cm, wordIds.begin(), wordIds.end());
	case Segmentation::one_pre:
		return model.template getScore<Segmentation::one_pre>(cm, wordIds.begin(), wordIds.end());
	case Segmentation::one_suc:
		return model.template getScore<Segmentation::one_suc>(cm, wordIds.begin(), wordIds.end());
	case Segmentation::one_all:
		return model.template getScore<Segmentation::one_all>(cm, wordIds.begin(), wordIds.end());
	case Segmentation::one_set:
		return model.template getScore<Segmentation::one_set>(cm, wordIds.begin(), wordIds.end());
	default:
		throw py::ValueError{ "invalid Segmentation `seg`" };
	}
}

void addCoherenceTypes(py::Module& module)
{
	module.addType(
		py::define<CoherenceObject>("tomotopy._Coherence", "_Coherence", Py_TPFLAGS_BASETYPE)
			.method<&CoherenceObject::getScore>("get_score")
	);
}
