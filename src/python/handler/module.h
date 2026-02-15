#pragma once

#include <fstream>
#include <iostream>

#define USE_NUMPY
#ifdef _DEBUG
//#undef _DEBUG
#define DEBUG_LOG(t) do{ cerr << t << endl; }while(0)
#include "PyUtils.h"
//#define _DEBUG
#else 
#define DEBUG_LOG(t)
#include "PyUtils.h"
#endif

#include "../../TopicModel/TopicModel.hpp"
#include "../../TopicModel/LDA.h"
#include "../../TopicModel/DMR.h"
#include "../../TopicModel/GDMR.h"
#include "../../TopicModel/PA.h"
#include "../../TopicModel/HPA.h"
#include "../../TopicModel/MGLDA.h"
#include "../../TopicModel/HDP.h"
#include "../../TopicModel/HLDA.h"
#include "../../TopicModel/CT.h"
#include "../../TopicModel/SLDA.h"
#include "../../TopicModel/LLDA.h"
#include "../../TopicModel/PLDA.h"
#include "../../TopicModel/DT.h"
#include "../../TopicModel/PT.h"
#include "../../Utils/serializer.hpp"
#include "../../Utils/Mmap.h"
#include "docs.h"

void char2Byte(const std::string& str, std::vector<uint32_t>& startPos, std::vector<uint16_t>& length);

void char2Byte(const char* begin, const char* end, std::vector<uint32_t> & startPos, std::vector<uint16_t> & length);

struct VocabObject;
struct CorpusObject;
struct DocumentObject;

struct TopicModelObject
{
	std::shared_ptr<tomoto::ITopicModel> inst;
	bool isPrepared = false, seedGiven = false;
	size_t minWordCnt = 0, minWordDf = 0;
	size_t removeTopWord = 0;

	template<class Ty>
	Ty* getInst() const
	{
		if (!inst) throw py::RuntimeError{ "inst is null" };
		return static_cast<Ty*>(inst.get());
	}

	template<class Ty>
	Ty* getInstDynamic() const
	{
		if (!inst) throw py::RuntimeError{ "inst is null" };
		return dynamic_cast<Ty*>(inst.get());
	}

	virtual ~TopicModelObject() = default;
	virtual PyObject* getObject() const = 0;
	virtual tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const;

	std::vector<size_t> insertCorpus(PyObject* corpusObj, PyObject* transform);
	py::UniqueCObj<CorpusObject> makeCorpus(PyObject* _corpus, PyObject* transform) const;
};

struct LDAModelObject : public py::CObject<LDAModelObject>, public TopicModelObject
{
	LDAModelObject() = default;
	LDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* alpha, float eta, PyObject* seed,
		PyObject* corpus, PyObject* transform);

	PyObject* getObject() const override { return (PyObject*)this; }

	py::UniqueObj addCorpus(PyObject* corpus, PyObject* transform);

	std::optional<size_t> addDoc(PyObject* words, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words);

	void setWordPrior(const std::string& word, PyObject* prior);
	std::vector<tomoto::Float> getWordPrior(const std::string& word) const;
	void train(size_t iteration, size_t workers, size_t ps, bool freezeTopics, size_t callbackInterval, PyObject* callback);
	py::UniqueObj getTopicWords(size_t topicId, size_t topN, bool returnId = false) const;
	std::vector<tomoto::Float> getTopicWordDist(size_t topicId, bool normalize = true) const;
	py::UniqueObj infer(PyObject* doc, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const;
	void save(const std::string& filename, const std::vector<uint8_t>& extraData, bool full = true);
	py::UniqueObj saves(const std::vector<uint8_t>& extraData, bool full = true);
	void updateVocab(const std::vector<std::string>& newVocabs);

	py::UniqueCObj<CorpusObject> getDocs() const;
	py::UniqueCObj<VocabObject> getVocabs() const;
	py::UniqueCObj<VocabObject> getUsedVocabs() const;
	py::UniqueObj getUsedVocabCf() const;
	std::vector<double> getUsedVocabWeightedCf() const;
	py::UniqueObj getUsedVocabDf() const;
	std::vector<uint64_t> getCountByTopics() const;
	std::vector<float> getAlpha() const;
	std::vector<std::string> getRemovedTopWords() const;
	py::UniqueObj getWordForms(size_t idx = -1) const;
	py::UniqueObj getHash() const;
	py::UniqueCObj<LDAModelObject> copy(PyObject* cls) const;
	static std::pair<py::UniqueCObj<LDAModelObject>, std::vector<uint8_t>> load(PyObject* cls, const std::string& filename);
	static std::pair<py::UniqueCObj<LDAModelObject>, std::vector<uint8_t>> loads(PyObject* cls, const std::vector<uint8_t>& data);
};

struct DMRModelObject : public py::CObject<DMRModelObject>, public TopicModelObject
{
	DMRModelObject() = default;
	DMRModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* alpha, float eta, float sigma, float alphaEps,
		PyObject* seed, PyObject* corpus, PyObject* transform);

	PyObject* getObject() const override { return (PyObject*)this; }
	
	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;

	std::optional<size_t> addDoc(PyObject* words, const std::string& metadata, PyObject* multiMetadata, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, const std::string& metadata, PyObject* multiMetadata);
	std::vector<float> getTopicPrior(const std::string& metadata, PyObject* multiMetadata, bool raw = false) const;
	py::UniqueCObj<VocabObject> getMetadataDict() const;
	py::UniqueCObj<VocabObject> getMultiMetadataDict() const;
	py::UniqueObj getLambda() const;
	py::UniqueObj getLambdaV2() const;
	py::UniqueObj getAlpha() const;
};

struct GDMRModelObject : public py::CObject<GDMRModelObject>, public TopicModelObject
{
	GDMRModelObject() = default;
	GDMRModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* degrees, PyObject* alpha, float eta, float sigma, float sigma0, float alphaEps,
		float orderDecay, PyObject* range,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;

	std::optional<size_t> addDoc(PyObject* words, PyObject* numericMetadata, const std::string& metadata, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, PyObject* numericMetadata, const std::string& metadata);
	std::vector<float> tdf(PyObject* numericMetadata, const std::string& metadata, PyObject* multiMetadata, bool normalize = true) const;
	py::UniqueObj tdfLinspace(PyObject* numericMetadataStart, PyObject* numericMetadataEnd, PyObject* num, const std::string& metadata, PyObject* multiMetadata, bool endpoint = true, bool normalize = true) const;

	std::vector<std::pair<float, float>> getMetadataRange() const;
	void getTopicPrior() const;
};

struct PAModelObject : public py::CObject<PAModelObject>, public TopicModelObject
{
	PAModelObject() = default;
	PAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k1, size_t k2, 
		PyObject* alpha, PyObject* subAlpha, float eta,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	std::vector<float> getSubTopicPrior(size_t superTopicId, bool normalize = true) const;
	std::vector<std::pair<tomoto::Tid, tomoto::Float>> getSubTopics(size_t superTopicId, size_t topN = 10) const;
	std::vector<std::pair<std::string, tomoto::Float>> getTopicWords(size_t subTopicId, size_t topN = 10) const;
	std::vector<float> getTopicWordDist(size_t subTopicId, bool normalize = true) const;

	py::UniqueObj infer(PyObject* doc, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const;
	std::vector<size_t> getCountBySuperTopic() const;
	py::UniqueObj getSubAlpha() const;
};

struct HPAModelObject : public py::CObject<HPAModelObject>, public TopicModelObject
{
	HPAModelObject() = default;
	HPAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k1, size_t k2,
		PyObject* alpha, PyObject* subAlpha, float eta,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	std::vector<std::pair<std::string, tomoto::Float>> getTopicWords(size_t topicId, size_t topN = 10) const;
	std::vector<float> getTopicWordDist(size_t topicId, bool normalize = true) const;
	py::UniqueObj infer(PyObject* doc, size_t iteration, float tolerance, size_t workers, tomoto::ParallelScheme ps, bool together, PyObject* transform) const;
	py::UniqueObj getAlpha() const;
	py::UniqueObj getSubAlpha() const;
};

struct MGLDAModelObject : public py::CObject<MGLDAModelObject>, public TopicModelObject
{
	MGLDAModelObject() = default;
	MGLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, size_t kL, size_t t, float alpha, float alphaL, float alphaMG, float alphaML, 
		float eta, float etaL, float gamma,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;

	std::optional<size_t> addDoc(PyObject* words, const std::string& delimiter, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, const std::string& delimiter);

	std::vector<std::pair<std::string, tomoto::Float>> getTopicWords(size_t topicId, size_t topN = 10) const;
	std::vector<float> getTopicWordDist(size_t topicId, bool normalize = true) const;
};

struct HDPModelObject : public py::CObject<HDPModelObject>, public TopicModelObject
{
	HDPModelObject() = default;
	HDPModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t initialK, float alpha, float eta, float gamma,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	bool isLiveTopic(size_t topicId) const;
	std::pair<py::UniqueCObj<LDAModelObject>, py::UniqueObj> convertToLDA(PyObject* LDAType, float topicThreshold) const;
	std::vector<int32_t> purgeDeadTopics();

	float getAlpha() const;
};

struct HLDAModelObject : public py::CObject<HLDAModelObject>, public TopicModelObject
{
	HLDAModelObject() = default;
	HLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t depth, PyObject* alpha, float eta, float gamma,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }
	
	std::vector<float> getAlpha() const;
	
	template<auto memFn>
	py::UniqueObj getTopicInfo(size_t topicId) const 
	{
		auto* inst = getInst<tomoto::IHLDAModel>();
		if (topicId >= inst->getK()) throw py::ValueError{ "must topic_id < K" };
		if (!isPrepared) throw py::RuntimeError{ "train() should be called first" };
		return py::buildPyValue((inst->*memFn)(topicId));
	}
};

struct CTModelObject : public py::CObject<CTModelObject>, public TopicModelObject
{
	CTModelObject() = default;
	CTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* alpha, float eta,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	py::UniqueObj getCorrelations(PyObject* topicId) const;
	py::UniqueObj getPriorCov() const;
};

struct SLDAModelObject : public py::CObject<SLDAModelObject>, public TopicModelObject
{
	SLDAModelObject() = default;
	SLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* vars, PyObject* alpha, float eta,
		PyObject* mu, PyObject* nuSq, PyObject* glmCoef,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;
	std::optional<size_t> addDoc(PyObject* words, PyObject* y, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, PyObject* y);
	py::UniqueObj getRegressionCoef(PyObject* varId) const;
	std::string getTypeOfVar(size_t varId) const;
	py::UniqueObj estimateVars(PyObject* doc) const;
};

struct LLDAModelObject : public py::CObject<LLDAModelObject>, public TopicModelObject
{
	LLDAModelObject() = default;
	LLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, PyObject* alpha, float eta, 
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;
	std::optional<size_t> addDoc(PyObject* words, PyObject* labels, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, PyObject* labels);
	py::UniqueCObj<VocabObject> getTopicLabelDict() const;
};

struct PLDAModelObject : public py::CObject<PLDAModelObject>, public TopicModelObject
{
	PLDAModelObject() = default;
	PLDAModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t latentTopics, size_t topicsPerLabel, PyObject* alpha, float eta, float sigma,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;
	std::optional<size_t> addDoc(PyObject* words, PyObject* labels, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, PyObject* labels);
	py::UniqueCObj<VocabObject> getTopicLabelDict() const;
};

struct DTModelObject : public py::CObject<DTModelObject>, public TopicModelObject
{
	DTModelObject() = default;
	DTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, size_t t, float alphaVar, float etaVar, float phiVar,
		float lrA, float lrB, float lrC,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }

	tomoto::RawDoc::MiscType convertMisc(const tomoto::RawDoc::MiscType& o) const override;
	std::optional<size_t> addDoc(PyObject* words, size_t timepoint, bool ignoreEmptyWords = true);
	py::UniqueCObj<DocumentObject> makeDoc(PyObject* words, size_t timepoint);
	std::vector<float> getAlpha2(size_t timepoint) const;
	std::vector<float> getPhi(size_t timepoint, size_t topicId) const;
	std::vector<std::pair<std::string, tomoto::Float>> getTopicWords(size_t topicId, size_t timepoint, size_t topN = 10) const;
	std::vector<float> getTopicWordDist(size_t topicId, size_t timepoint, bool normalize = true) const;
	py::UniqueObj getCountByTopic() const;
	py::UniqueObj getAlpha() const;
};

struct PTModelObject : public py::CObject<PTModelObject>, public TopicModelObject
{
	PTModelObject() = default;
	PTModelObject(size_t tw, size_t minCnt, size_t minDf, size_t rmTop,
		size_t k, size_t p, PyObject* alpha, float eta,
		PyObject* seed, PyObject* corpus, PyObject* transform);
	PyObject* getObject() const override { return (PyObject*)this; }
};

template<typename Ty, typename FailMsg>
inline std::vector<Ty> broadcastObj(PyObject* obj, size_t k, FailMsg&& msg)
{
	try
	{
		std::vector<Ty> ret;
		try
		{
			ret = py::toCpp<std::vector<Ty>>(obj);
			if (ret.size() != k) throw py::ConversionFail{ "" };
		}
		catch (const py::ConversionFail&)
		{
			PyErr_Clear();
			ret.emplace_back(py::toCpp<Ty>(obj));
		}
		return ret;
	}
	catch (const py::ConversionFail&)
	{
		throw py::ConversionFail{ std::forward<FailMsg>(msg) };
	}
}

namespace py
{
	struct RawDocVarToPy
	{
		template<typename _Ty>
		py::UniqueObj operator()(const _Ty& s)
		{
			return buildPyValue(s);
		}

		py::UniqueObj operator()(const std::shared_ptr<void>& s)
		{
			if (s)
			{
				py::UniqueObj ret{ (PyObject*)s.get() };
				Py_INCREF(ret.get());
				return ret;
			}
			return {};
		}
	};

	template<>
	struct ValueBuilder<tomoto::RawDoc::Var>
	{
		py::UniqueObj operator()(const tomoto::RawDoc::Var& v)
		{
			RawDocVarToPy visitor;
			return std::move(std::visit(visitor, v));
		}

		bool _toCpp(PyObject* obj, tomoto::RawDoc::Var& out)
		{
			Py_INCREF(obj);
			out = std::shared_ptr<void>{ obj, [](void* p)
			{
				Py_XDECREF(p);
			} };
			return true;
		}
	};
}

template<typename _Ty>
_Ty getValueFromMisc(const char* key, const tomoto::RawDoc::MiscType& misc, const char* failMsg)
{
	auto it = misc.find(key);
	if (it == misc.end()) throw std::runtime_error{ failMsg + std::string{ " (the required value was not given)" } };
	auto obj = (PyObject*)std::get<std::shared_ptr<void>>(it->second).get();
	_Ty ret;
	if (!py::toCpp(obj, ret))
	{
		throw std::runtime_error{ failMsg + (" (given " + py::repr(obj) + ")") };
	}
	return ret;
}

template<typename _Ty>
_Ty getValueFromMiscDefault(const char* key, const tomoto::RawDoc::MiscType& misc, const char* failMsg, const _Ty& def = {})
{
	auto it = misc.find(key);
	if (it == misc.end()) return def;
	auto obj = (PyObject*)std::get<std::shared_ptr<void>>(it->second).get();
	_Ty ret;
	if (!py::toCpp(obj, ret))
	{
		throw std::runtime_error{ failMsg + (" (given " + py::repr(obj) + ")") };
	}
	return ret;
}

inline tomoto::RawDoc buildRawDoc(PyObject* words)
{
	tomoto::RawDoc raw;
	if (!py::toCpp<std::vector<std::string>>(words, raw.rawWords))
	{
		throw py::ValueError{ "`words` must be an iterable of str." };
	}
	return raw;
}
