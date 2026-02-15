#include "../../TopicModel/PLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType PLDAModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["labels"] = getValueFromMiscDefault<vector<string>>("labels", o, "`PLDAModel` requires a `labels` value in `Iterable[str]` type.");
	return ret;
}

PLDAModelObject::PLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t latentTopics, size_t topicsPerLabel, PyObject* alpha, float eta, float sigma,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::PLDAArgs mArgs;
	mArgs.numLatentTopics = latentTopics;
	mArgs.numTopicsPerLabel = topicsPerLabel;
	if (alpha)
	{
		mArgs.alpha = broadcastObj<tomoto::Float>(alpha, mArgs.k,
			[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
		);
	}
	mArgs.eta = eta;
	if (seed && !py::toCpp<size_t>(seed, mArgs.seed))
	{
		throw invalid_argument{ "`seed` must be an integer or None." };
	}

	inst = tomoto::IPLDAModel::create((tomoto::TermWeight)tw, mArgs);
	if (!inst) throw py::ValueError{ "unknown tw value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

std::optional<size_t> PLDAModelObject::addDoc(PyObject* words, PyObject* labels, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::IPLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	if (labels)
	{
		if (PyUnicode_Check(labels))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`labels` should be an iterable of str.", 1)) throw py::ExcPropagation{};
		}
		vector<string> labelVec;
		if (!py::toCpp(labels, labelVec))
		{
			throw invalid_argument{ "`labels` must be an iterable of str." };
		}
		raw.misc["labels"] = labelVec;
	}
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

py::UniqueCObj<DocumentObject> PLDAModelObject::makeDoc(PyObject* words, PyObject* labels)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::IPLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);

	if (labels)
	{
		if (PyUnicode_Check(labels))
		{
			if (PyErr_WarnEx(PyExc_RuntimeWarning, "`labels` should be an iterable of str.", 1)) throw py::ExcPropagation{};
		}
		vector<string> labelVec;
		if (!py::toCpp(labels, labelVec))
		{
			throw invalid_argument{ "`labels` must be an iterable of str." };
		}
		raw.misc["labels"] = labelVec;
	}
	auto doc = inst->makeDoc(raw);
	py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, (PyObject*)this, nullptr) };
	auto ret = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, corpus.get(), nullptr) };
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

py::UniqueCObj<VocabObject> PLDAModelObject::getTopicLabelDict() const
{
	auto* inst = getInst<tomoto::IPLDAModel>();
	auto ret = py::makeNewObject<VocabObject>();
	ret->dep = py::UniqueObj{ (PyObject*)this };
	Py_INCREF(ret->dep);
	ret->vocabs = const_cast<tomoto::Dictionary*>(&inst->getTopicLabelDict());
	ret->size = -1;
	return ret;
}
