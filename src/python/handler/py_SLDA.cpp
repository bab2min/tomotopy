#include "../../TopicModel/SLDA.h"

#include "module.h"
#include "utils.h"

using namespace std;

tomoto::RawDoc::MiscType SLDAModelObject::convertMisc(const tomoto::RawDoc::MiscType& o) const
{
	tomoto::RawDoc::MiscType ret;
	ret["y"] = getValueFromMiscDefault<vector<float>>("y", o, "`SLDAModel` requires a `y` value in `Iterable[float]` type.");
	return ret;
}

SLDAModelObject::SLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
	size_t k, PyObject* vars, PyObject* alpha, float eta,
	PyObject* mu, PyObject* nuSq, PyObject* glmCoef,
	PyObject* seed, PyObject* corpus, PyObject* transform)
{
	tomoto::SLDAArgs margs;
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

	vector<string> varTypeStrs;
	if (vars && !py::toCpp<vector<string>>(vars, varTypeStrs))
	{
		throw invalid_argument{ "`vars` must be an iterable of str." };
	}
	for (auto& s : varTypeStrs)
	{
		tomoto::ISLDAModel::GLM t;
		if (s == "l") t = tomoto::ISLDAModel::GLM::linear;
		else if (s == "b") t = tomoto::ISLDAModel::GLM::binary_logistic;
		else throw py::ValueError{ "Unknown var type '" + s + "'" };
		margs.vars.emplace_back(t);
	}

	float fTemp;
	if (mu)
	{
		if (py::toCpp<float>(mu, fTemp))
		{
			margs.mu.resize(varTypeStrs.size(), fTemp);
		}
		else if (py::toCpp<vector<tomoto::Float>>(mu, margs.mu))
		{

		}
		else
		{
			throw invalid_argument{ "`mu` must be a float or an iterable of float with length same as `vars`." };
		}
	}

	if (nuSq)
	{
		if (py::toCpp<float>(nuSq, fTemp))
		{
			margs.nuSq.resize(varTypeStrs.size(), fTemp);
		}
		else if (py::toCpp<vector<tomoto::Float>>(nuSq, margs.nuSq))
		{
		}
		else
		{
			throw invalid_argument{ "`nu_sq` must be a float or an iterable of float with length same as `vars`." };
		}
	}

	if (glmCoef)
	{
		if (py::toCpp<float>(glmCoef, fTemp))
		{
			margs.glmParam.resize(varTypeStrs.size(), fTemp);
		}
		else if (py::toCpp<vector<tomoto::Float>>(glmCoef, margs.glmParam))
		{
		}
		else
		{
			throw invalid_argument{ "`glm_param` must be a float or an iterable of float with length same as `vars`." };
		}
	}

	inst = tomoto::ISLDAModel::create((tomoto::TermWeight)tw, margs);
	if (!inst) throw py::ValueError{ "unknown `tw` value" };
	isPrepared = false;
	seedGiven = !!seed;
	minWordCnt = minCnt;
	minWordDf = minDf;
	removeTopWord = rmTop;

	insertCorpus(corpus, transform);
}

std::optional<size_t> SLDAModelObject::addDoc(PyObject* words, PyObject* y, bool ignoreEmptyWords)
{
	if (isPrepared) throw py::RuntimeError{ "cannot add_doc() after train()" };
	auto* inst = getInst<tomoto::ISLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);

	if (y)
	{
		vector<tomoto::Float> yVec;
		if (!py::toCpp(y, yVec))
		{
			throw py::ValueError{ "`y` must be an iterable of float." };
		}
		raw.misc["y"] = yVec;
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

py::UniqueCObj<DocumentObject> SLDAModelObject::makeDoc(PyObject* words, PyObject* y)
{
	if (!isPrepared) throw py::RuntimeError{ "`train()` should be called before `make_doc()`." };
	auto* inst = getInst<tomoto::ISLDAModel>();
	if (PyUnicode_Check(words))
	{
		if (PyErr_WarnEx(PyExc_RuntimeWarning, "`words` should be an iterable of str.", 1)) throw py::ExcPropagation{};
	}
	tomoto::RawDoc raw = buildRawDoc(words);

	if (y)
	{
		vector<tomoto::Float> yVec;
		if (!py::toCpp(y, yVec))
		{
			throw py::ValueError{ "`y` must be an iterable of float." };
		}
		raw.misc["y"] = yVec;
	}
	auto doc = inst->makeDoc(raw);
	py::UniqueObj corpus{ PyObject_CallFunctionObjArgs((PyObject*)py::Type<CorpusObject>, (PyObject*)this, nullptr) };
	auto ret = py::UniqueCObj<DocumentObject>{ (DocumentObject*)PyObject_CallFunctionObjArgs((PyObject*)py::Type<DocumentObject>, corpus.get(), nullptr) };
	ret->doc = doc.release();
	ret->owner = true;
	return ret;
}

py::UniqueObj SLDAModelObject::getRegressionCoef(PyObject* varId) const
{
	auto* inst = getInst<tomoto::ISLDAModel>();
	if (!varId || varId == Py_None)
	{
		float* ptr;
		py::UniqueObj ret = py::newEmptyArray(ptr, inst->getF(), inst->getK());
		for (size_t i = 0; i < inst->getF(); ++i)
		{
			auto l = inst->getRegressionCoef(i);
			memcpy(ptr + i * inst->getK(), l.data(), sizeof(float) * inst->getK());
		}
		return ret;
	}

	size_t varIdVal;
	if (!py::toCpp(varId, varIdVal)) throw py::ValueError{ "`var_id` must be an integer or None." };
	if (varIdVal >= inst->getF()) throw py::ValueError{ "`var_id` must be < `f`" };
	return py::buildPyValue(inst->getRegressionCoef(varIdVal));
}

std::string SLDAModelObject::getTypeOfVar(size_t varId) const
{
	auto* inst = getInst<tomoto::ISLDAModel>();
	if (varId >= inst->getF()) throw py::ValueError{ "`var_id` must be < `f`" };
	return std::string{ "l\0b" + (size_t)inst->getTypeOfVar(varId) * 2 };
}

py::UniqueObj SLDAModelObject::estimateVars(PyObject* docObj) const
{
	auto* inst = getInst<tomoto::ISLDAModel>();
	try
	{
		if (!PyObject_TypeCheck(docObj, py::Type<DocumentObject>)) throw py::ConversionFail{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
		auto* doc = (DocumentObject*)docObj;
		if ((SLDAModelObject*)doc->corpus->tm.get() != this) throw py::ConversionFail{ "`doc` was from another model, not fit to this model" };

		return py::buildPyValue(inst->estimateVars(doc->getBoundDoc()));
	}
	catch (const py::ConversionFail&)
	{
		PyErr_Clear();
	}

	py::UniqueObj iter = py::UniqueObj{ PyObject_GetIter(docObj) };
	py::UniqueObj nextDoc;
	std::vector<const tomoto::DocumentBase*> docs;
	while ((nextDoc = py::UniqueObj{ PyIter_Next(iter.get()) }))
	{
		if (!PyObject_TypeCheck(nextDoc, py::Type<DocumentObject>)) throw py::ConversionFail{ "`doc` must be tomotopy.Document or list of tomotopy.Document" };
		auto* doc = (DocumentObject*)nextDoc.get();
		if ((SLDAModelObject*)doc->corpus->tm.get() != this) throw py::ConversionFail{ "`doc` was from another model, not fit to this model" };
		docs.emplace_back(doc->getBoundDoc());
	}
	if (PyErr_Occurred()) throw py::ExcPropagation{};
	return py::buildPyValueTransform(docs.begin(), docs.end(), [&](const tomoto::DocumentBase* d)
	{
		return inst->estimateVars(d);
	});
}

py::UniqueObj DocumentObject::getY() const 
{
	if (corpus->isIndependent()) throw py::AttributeError{ "doc has no `" "y" "` field!" }; 
	if (!doc) throw py::RuntimeError{ "doc is null!" }; 
	if (auto ret = docVisit<tomoto::DocumentSLDA>(getBoundDoc(), [](auto* doc) 
	{ 
		return py::buildPyValue(doc->y); 
	})) return ret; 
	throw py::AttributeError{ "doc has no `" "y" "` field!" };
}
