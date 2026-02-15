#include "../../TopicModel/DMR.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType DMRModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["metadata"] = getValueFromMiscDefault<string>("metadata", o, "`DMRModel` needs a `metadata` value in `str` type.");
	ret["multi_metadata"] = getValueFromMiscDefault<vector<string>>("multi_metadata", o, "`DMRModel` needs a `multi_metadata` value in `List[str]` type.");
	return ret;
}

DMRModelObject::DMRModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, PyObject* alpha, float eta, float sigma, float alphaEps,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::DMRArgs margs;
	margs.k = k;
	if (alpha) margs.alpha = broadcastObj<tomoto::Float>(alpha, margs.k,
		[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
	);
	margs.eta = eta;
	margs.sigma = sigma;
	margs.alphaEps = alphaEps;
	if (seed && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw invalid_argument{ "`seed` must be an integer or None." };
	}

	auto inst = tomoto::IDMRModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	this->inst = std::move(inst);
	this->isPrepared = false;
	this->seedGiven = !!seed;
	this->minWordCnt = minCnt;
	this->minWordDf = minDf;
	this->removeTopWord = rmTop;
	insertCorpus(corpus, transform);
}

std::optional<size_t> DMRModelObject::addDoc(PyObject* words, const std::string& metadata, PyObject* multiMetadata, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::IDMRModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	if (multiMetadata && PyUnicode_Check(multiMetadata))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["metadata"] = metadata;
	if (multiMetadata)
	{
		vector<string> multiMetadataValue;
		if (!py::toCpp<vector<string>>(multiMetadata, multiMetadataValue))
		{
			throw py::ValueError{
				"`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"
			};
		}
		raw.misc["multi_metadata"] = move(multiMetadataValue);
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

py::UniqueCObj<DocumentObject> DMRModelObject::makeDoc(PyObject* words, const std::string& metadata, PyObject* multiMetadata)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::IDMRModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	if (multiMetadata && PyUnicode_Check(multiMetadata))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["metadata"] = metadata;
	if (multiMetadata)
	{
		vector<string> multiMetadataValue;
		if (!py::toCpp<vector<string>>(multiMetadata, multiMetadataValue))
		{
			throw py::ValueError{
				"`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"
			};
		}
		raw.misc["multi_metadata"] = move(multiMetadataValue);
	}
	auto doc = inst->makeDoc(raw);
	py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, (PyObject*)this, nullptr) };
	auto ret = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, corpus.get(), nullptr) };
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

std::vector<float> DMRModelObject::getTopicPrior(const std::string& metadata, PyObject* multiMetadata, bool raw) const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	if (multiMetadata && PyUnicode_Check(multiMetadata))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`multi_metadata` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}

	vector<string> multiMd;
	if (multiMetadata)
	{
		if (!py::toCpp<vector<string>>(multiMetadata, multiMd))
		{
			throw py::ValueError{
				"`multi_metadata` must be an instance of `List[str]` (but given " + py::repr(multiMetadata) + ")"
			};
		}
	}
	return inst->getTopicPrior(metadata, multiMd, raw);
}

py::UniqueCObj<VocabObject> DMRModelObject::getMetadataDict() const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	auto ret = py::makeNewObject<VocabObject>();
	ret->dep = py::UniqueObj{ (PyObject*)this };
	Py_INCREF(ret->dep);
	ret->vocabs = const_cast<tomoto::Dictionary*>(&inst->getMetadataDict());
	ret->size = -1;
	return ret;
}

py::UniqueCObj<VocabObject> DMRModelObject::getMultiMetadataDict() const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	auto ret = py::makeNewObject<VocabObject>();
	ret->dep = py::UniqueObj{ (PyObject*)this };
	Py_INCREF(ret->dep);
	ret->vocabs = const_cast<tomoto::Dictionary*>(&inst->getMultiMetadataDict());
	ret->size = -1;
	return ret;
}

py::UniqueObj DMRModelObject::getLambda() const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	float* ptr;
	const size_t fs = inst->getF() * inst->getMdVecSize();
	auto ret = py::newEmptyArray(ptr, inst->getK(), fs);
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		auto l = inst->getLambdaByTopic(i);
		memcpy(ptr + i * fs, l.data(), sizeof(float) * fs);
	}
	return ret;
}

py::UniqueObj DMRModelObject::getLambdaV2() const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	float* ptr;
	size_t fs = inst->getF() * inst->getMdVecSize();
	auto ret = py::newEmptyArray(ptr, inst->getK(), inst->getF(), inst->getMdVecSize());
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		auto l = inst->getLambdaByTopic(i);
		memcpy(ptr + i * fs, l.data(), sizeof(float) * fs);
	}
	return ret;
}

py::UniqueObj DMRModelObject::getAlpha() const
{
	auto* inst = getInst<tomoto::IDMRModel>();
	float* ptr;
	auto ret = py::newEmptyArray(ptr, inst->getK(), inst->getF());
	for (size_t i = 0; i < inst->getK(); ++i)
	{
		auto l = inst->getLambdaByTopic(i);
		Eigen::Map<Eigen::ArrayXf> ml{ l.data(), (Eigen::Index)l.size() };
		ml = ml.exp() + inst->getAlphaEps();
		memcpy(ptr + i * inst->getF(), l.data(), sizeof(float) * inst->getF());
	}
	return ret;
}

std::optional<std::string_view> DocumentObject::getMetadataFromDMR() const
{
	if (corpus->isIndependent()) return nullptr;
	if (!doc) throw py::RuntimeError{ "doc is null!" };
	auto inst = corpus->tm->getInst<tomoto::IDMRModel>();

	return docVisit<tomoto::DocumentDMR>(getBoundDoc(), [&](auto* doc)
	{
		return inst->getMetadataDict().toWord(doc->metadata);
	});
}

py::UniqueObj DocumentObject::getMultiMetadata() const
{
	if (!doc) throw py::RuntimeError{ "doc is null!" };
	auto inst = corpus->tm->getInst<tomoto::IDMRModel>();
		
	if(auto ret = docVisit<tomoto::DocumentDMR>(getBoundDoc(), [&](auto* doc)
	{
		return py::buildPyValueTransform(doc->multiMetadata.begin(), doc->multiMetadata.end(), [&](uint64_t x)
		{
			return inst->getMultiMetadataDict().toWord(x);
		});
	})) return ret;
	throw py::AttributeError{ "doc has no `multi_metadata` field!" };
}
