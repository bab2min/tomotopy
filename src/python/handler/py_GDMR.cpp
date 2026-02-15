#include "../../TopicModel/GDMR.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType GDMRModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["metadata"] = getValueFromMiscDefault<string>("metadata", o, 
		"Since version 0.11.0, `GDMRModel` requires a `metadata` value in `str` type. You can store numerical metadata to a `numeric_metadata` argument."
	);
	ret["numeric_metadata"] = getValueFromMiscDefault<vector<tomoto::Float>>("numeric_metadata", o, 
		"`GDMRModel` requires a `numeric_metadata` value in `Iterable[float]` type."
	);
	return ret;
}

GDMRModelObject::GDMRModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, PyObject* degrees, PyObject* alpha, float eta, float sigma, float sigma0, float alphaEps,
	float orderDecay, PyObject* range,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::GDMRArgs margs;
	if (alpha) margs.alpha = broadcastObj<tomoto::Float>(alpha, margs.k,
		[&]() { return "`alpha` must be an instance of `float` or `List[float]` with length `k` (given " + py::repr(alpha) + ")"; }
	);
	margs.eta = eta;
	margs.sigma = sigma;
	margs.sigma0 = sigma0;
	margs.alphaEps = alphaEps;
	margs.orderDecay = orderDecay;
	if (seed && !py::toCpp<size_t>(seed, margs.seed))
	{
		throw invalid_argument{ "`seed` must be an integer or None." };
	}

	if (degrees && !py::toCpp<vector<uint64_t>>(degrees, margs.degrees))
	{
		throw invalid_argument{ "`degrees` must be an iterable of int." };
	}

	auto inst = tomoto::IGDMRModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	this->inst = std::move(inst);
	this->isPrepared = false;
	this->seedGiven = !!seed;
	this->minWordCnt = minCnt;
	this->minWordDf = minDf;
	this->removeTopWord = rmTop;
	
	if (range && range != Py_None)
	{
		vector<tomoto::Float> vMin, vMax;
		py::UniqueObj rangeIter{ PyObject_GetIter(range) }, item;
		if (!rangeIter) throw py::ValueError{ "`metadata_range` must be a list of pairs." };
		while (item = py::UniqueObj{ PyIter_Next(rangeIter.get()) })
		{
			vector<tomoto::Float> r;
			if (!py::toCpp(item.get(), r))
			{
				throw py::ValueError{ "`metadata_range` must be a list of pairs." };
			}
			if (r.size() != 2) throw py::ValueError{ "`metadata_range` must be a list of pairs." };
			vMin.emplace_back(r[0]);
			vMax.emplace_back(r[1]);
		}
		if (vMin.size() != margs.degrees.size()) throw py::ValueError{ "`len(metadata_range)` must be equal to `len(degrees)`" };

		inst->setMdRange(vMin, vMax);
	}

	insertCorpus(corpus, transform);
}

std::optional<size_t> GDMRModelObject::addDoc(PyObject* words, PyObject* numericMetadata, const std::string& metadata, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::IGDMRModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}

	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["metadata"] = metadata;

	vector<tomoto::Float> nmd;
	if (!py::toCpp(numericMetadata, nmd)) 
	{
		throw py::ValueError{
			"`numeric_metadata` must be an iterable of float (but given " + py::repr(numericMetadata) + ")"
		};
	}
	for (auto x : nmd)
	{
		if (!isfinite(x)) throw py::ValueError{ "`numeric_metadata` has non-finite value (" + py::reprFromCpp(nmd) + ")." };
	}
	raw.misc["numeric_metadata"] = move(nmd);
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

py::UniqueCObj<DocumentObject> GDMRModelObject::makeDoc(PyObject* words, PyObject* numericMetadata, const std::string& metadata)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::IGDMRModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}

	tomoto::RawDoc raw = buildRawDoc(words);
	raw.misc["metadata"] = metadata;

	vector<tomoto::Float> nmd;
	if (!py::toCpp(numericMetadata, nmd))
	{
		throw py::ValueError{
			"`numeric_metadata` must be an iterable of float (but given " + py::repr(numericMetadata) + ")"
		};
	}
	for (auto x : nmd)
	{
		if (!isfinite(x)) throw py::ValueError{ "`numeric_metadata` has non-finite value (" + py::reprFromCpp(nmd) + ")." };
	}
	raw.misc["numeric_metadata"] = move(nmd);

	auto doc = inst->makeDoc(raw);
	py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, (PyObject*)this, nullptr) };
	auto ret = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, corpus.get(), nullptr) };
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

std::vector<float> GDMRModelObject::tdf(PyObject* numericMetadata, const std::string& metadata, PyObject* multiMetadata, bool normalize) const
{
	auto* inst = getInst<tomoto::IGDMRModel>();

	vector<tomoto::Float> nmd;
	if (!py::toCpp(numericMetadata, nmd))
	{
		throw py::ValueError{
			"`numeric_metadata` must be an iterable of float (but given " + py::repr(numericMetadata) + ")"
		};
	}
	if (nmd.size() != inst->getFs().size()) throw py::ValueError{ "`len(numeric_metadata)` must be equal to `len(degree).`" };
		
	try
	{
		return inst->getTDF(nmd.data(), metadata, {}, normalize);
	}
	catch (const tomoto::exc::InvalidArgument& e)
	{
		throw py::ValueError{ e.what() };
	}
}

py::UniqueObj GDMRModelObject::tdfLinspace(PyObject* numericMetadataStart, PyObject* numericMetadataEnd, PyObject* numericMetadataNum, const std::string& metadata, PyObject* multiMetadata, bool endpoint, bool normalize) const
{
	auto* inst = getInst<tomoto::IGDMRModel>();

	vector<tomoto::Float> start, stop;
	if(!py::toCpp(numericMetadataStart, start))
	{
		throw py::ValueError{
			"`metadata_start` must be an iterable of float (but given " + py::repr(numericMetadataStart) + ")"
		};
	}
	if (!py::toCpp(numericMetadataEnd, stop))
	{
		throw py::ValueError{
			"`metadata_stop` must be an iterable of float (but given " + py::repr(numericMetadataEnd) + ")"
		};
	}

	vector<std::ptrdiff_t> num;
	if (!py::toCpp(numericMetadataNum, num))
	{
		throw py::ValueError{
			"`num` must be an iterable of int (but given " + py::repr(numericMetadataNum) + ")"
		};
	}

	if (start.size() != inst->getFs().size()) throw py::ValueError{ "`len(metadata_start)` must be equal to `len(degree).`" };
	if (stop.size() != inst->getFs().size()) throw py::ValueError{ "`len(metadata_stop)` must be equal to `len(degree).`" };
	if (num.size() != inst->getFs().size()) throw py::ValueError{ "`len(num)` must be equal to `len(degree).`" };

	std::ptrdiff_t tot = 1;
	for (auto& v : num)
	{
		if (v <= 0) v = 1;
		tot *= v;
	}

	Eigen::MatrixXf mds{ (Eigen::Index)num.size(), (Eigen::Index)tot };
	vector<std::ptrdiff_t> idcs(num.size());
	for (size_t i = 0; i < tot; ++i)
	{
		for (size_t j = 0; j < num.size(); ++j)
		{
			mds(j, i) = start[j] + (stop[j] - start[j]) * idcs[j] / (endpoint ? max(num[j] - 1, (std::ptrdiff_t)1) : num[j]);
		}

		idcs.back()++;
		for (int j = idcs.size() - 1; j >= 0; --j)
		{
			if (idcs[j] >= num[j])
			{
				idcs[j] = 0;
				if (j) idcs[j - 1]++;
			}
			else break;
		}
	}

	try
	{
		auto computed = inst->getTDFBatch(mds.data(), metadata, {}, num.size(), tot, normalize);
		num.emplace_back(inst->getK());
		float* ptr;
		auto ret = py::newEmptyArrayFromDim(ptr, num.size(), num.data());
		memcpy(ptr, computed.data(), sizeof(float) * computed.size());
		return ret;
	}
	catch (const tomoto::exc::InvalidArgument& e)
	{
		throw py::ValueError{ e.what() };
	}
}

std::vector<std::pair<float, float>> GDMRModelObject::getMetadataRange() const
{
	auto* inst = getInst<tomoto::IGDMRModel>();
	vector<float> vMin, vMax;
	inst->getMdRange(vMin, vMax);
	vector<pair<float, float>> ret;
	for (size_t i = 0; i < vMin.size(); ++i)
	{
		ret.emplace_back(vMin[i], vMax[i]);
	}
	return ret;
}

void GDMRModelObject::getTopicPrior() const
{
	throw py::RuntimeError{ "GDMRModel doesn't support get_topic_prior(). Use tdf() instead." };
}

std::optional<std::vector<tomoto::Float>> DocumentObject::getNumericMetadata() const
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `numericMetadata` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentGDMR>(getBoundDoc(), [](auto* doc) -> std::optional<std::vector<tomoto::Float>>
	{
		return doc->metadataOrg; 
	})) return ret; 
	throw py::AttributeError{ "doc has no `numericMetadata` field!" }; 
}
