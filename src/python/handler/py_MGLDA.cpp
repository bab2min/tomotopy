#include "../../TopicModel/MGLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType MGLDAModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["delimiter"] = getValueFromMiscDefault<string>("delimiter", o, "`MGLDAModel` requires a `delimiter` value in `str` type.", ".");
	return ret;
}

MGLDAModelObject::MGLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, size_t kL, size_t t, float alpha, float alphaL, float alphaMG, float alphaML,
	float eta, float etaL, float gamma,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::MGLDAArgs margs;
	margs.alpha[0] = alpha;
	margs.alphaL[0] = alphaL;
	margs.alphaMG = alphaMG;
	margs.alphaML = alphaML;
	margs.eta = eta;
	margs.etaL = etaL;
	margs.gamma = gamma;
	if (seed && seed != Py_None && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw py::ValueError{ "`seed` must be an integer or None." };
	}

	inst = tomoto::IMGLDAModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;
	
	insertCorpus(corpus, transform);
}

std::optional<size_t> MGLDAModelObject::addDoc(PyObject* words, const std::string& delimiter, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::IMGLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["delimiter"] = delimiter;
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

py::UniqueCObj<DocumentObject> MGLDAModelObject::makeDoc(PyObject* words, const std::string& delimiter)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::IMGLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["delimiter"] = delimiter;
	auto doc = inst->makeDoc(raw);
	py::UniqueCObj<CorpusObject> corpus{ (CorpusObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, Py_None, getObject(), nullptr) };
	auto ret = py::makeNewObject<DocumentObject>(getDocumentCls());
	ret->corpus = corpus.copy();
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

std::vector<std::pair<std::string, tomoto::Float>> MGLDAModelObject::getTopicWords(size_t topicId, size_t topN) const
{
	auto* inst = getInst<tomoto::IMGLDAModel>();
	if (topicId >= inst->getK() + inst->getKL()) throw py::ValueError{ "must topic_id < KG + KL" };

	return inst->getWordsByTopicSorted(topicId, topN);
}

std::vector<float> MGLDAModelObject::getTopicWordDist(size_t topicId, bool normalize) const
{
	auto* inst = getInst<tomoto::IMGLDAModel>();
	if (topicId >= inst->getK() + inst->getKL()) throw py::ValueError{ "must topic_id < KG + KL" };

	return inst->getWidsByTopic(topicId, normalize);
}

py::UniqueObj DocumentObject::getWindows() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `Vs` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentMGLDA>(getBoundDoc(), [](auto* doc) 
	{
		return buildPyValueReorder(doc->Vs, doc->wOrder); 
	})) return ret;
	throw py::AttributeError{ "doc has no `Vs` field!" };
}
