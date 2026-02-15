#define MAIN_MODULE
#include "module.h"
#include "label.h"
#include "utils.h"
#include "coherence.h"
#include "PyUtils.h"

using namespace std;

#ifdef TOMOTOPY_ISA
#define TO_STR(name) #name
#define TO_STR_WRAP(name) TO_STR(name)
#define TOMOTOPY_ISA_STR TO_STR_WRAP(TOMOTOPY_ISA)
static const char* isa_str = TOMOTOPY_ISA_STR;
#else
static const char* isa_str = "none";
#endif

void char2Byte(const char* strBegin, const char* strEnd, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	if (strBegin == strEnd) return;
	vector<size_t> charPos;
	auto it = strBegin;
	for (; it != strEnd; )
	{
		charPos.emplace_back(it - strBegin);
		uint8_t c = *it;
		if ((c & 0xF8) == 0xF0)
		{
			it += 4;
		}
		else if ((c & 0xF0) == 0xE0)
		{
			it += 3;
		}
		else if ((c & 0xE0) == 0xC0)
		{
			it += 2;
		}
		else if ((c & 0x80))
		{
			throw std::runtime_error{ "utf-8 decoding error" };
		}
		else it += 1;
	}
	charPos.emplace_back(strEnd - strBegin);

	for (size_t i = 0; i < startPos.size(); ++i)
	{
		size_t s = startPos[i], e = (size_t)startPos[i] + length[i];
		startPos[i] = charPos[s];
		length[i] = charPos[e] - charPos[s];
	}
}

void char2Byte(const string& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return char2Byte(&str[0], &str[0] + str.size(), startPos, length);
}

void char2Byte(const tomoto::SharedString& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return char2Byte(str.begin(), str.end(), startPos, length);
}

void byte2Char(const char* strBegin, const char* strEnd, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	if (strBegin == strEnd) return;
	vector<size_t> charPos;
	auto it = strBegin;
	for (; it != strEnd; )
	{
		charPos.emplace_back(it - strBegin);
		uint8_t c = *it;
		if ((c & 0xF8) == 0xF0)
		{
			it += 4;
		}
		else if ((c & 0xF0) == 0xE0)
		{
			it += 3;
		}
		else if ((c & 0xE0) == 0xC0)
		{
			it += 2;
		}
		else if ((c & 0x80))
		{
			throw std::runtime_error{ "utf-8 decoding error" };
		}
		else it += 1;
	}
	charPos.emplace_back(strEnd - strBegin);

	for (size_t i = 0; i < startPos.size(); ++i)
	{
		size_t s = startPos[i], e = (size_t)startPos[i] + length[i];
		startPos[i] = std::lower_bound(charPos.begin(), charPos.end(), s) - charPos.begin();
		length[i] = std::lower_bound(charPos.begin(), charPos.end(), e) - charPos.begin() - startPos[i];
	}
}

void byte2Char(const string& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return byte2Char(&str[0], &str[0] + str.size(), startPos, length);
}

void byte2Char(const tomoto::SharedString& str, vector<uint32_t>& startPos, vector<uint16_t>& length)
{
	return byte2Char(str.begin(), str.end(), startPos, length);
}

PyMODINIT_FUNC MODULE_NAME()
{
	py::Module module{ "tomotopy", "Tomoto Module for Python"};
	auto moduleObj = module.init(
		py::define<LDAModelObject>("tomotopy.LDAModel", "_LDAModel", Py_TPFLAGS_BASETYPE)
		.method<&LDAModelObject::addDoc>("add_doc")
		.method<&LDAModelObject::addCorpus>("add_corpus")
		.method<&LDAModelObject::makeDoc>("make_doc")
		.method<&LDAModelObject::setWordPrior>("set_word_prior")
		.method<&LDAModelObject::getWordPrior>("get_word_prior")
		.method<&LDAModelObject::train>("train")
		.method<&LDAModelObject::getCountByTopics>("get_count_by_topics")
		.method<&LDAModelObject::getTopicWords>("get_topic_words")
		.method<&LDAModelObject::getTopicWordDist>("get_topic_word_dist")
		.method<&LDAModelObject::infer>("infer")
		.method<&LDAModelObject::save>("save")
		.method<&LDAModelObject::saves>("saves")
		.staticMethod<&LDAModelObject::load>("load")
		.staticMethod<&LDAModelObject::loads>("loads")
		.method<&LDAModelObject::copy>("copy")
		.method<&LDAModelObject::updateVocab>("_update_vocab")
		.method<&LDAModelObject::getWordForms>("get_word_forms")
		.method<&LDAModelObject::getHash>("get_hash")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getK>("k")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getEta>("eta")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getPerplexity>("perplexity")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getLLPerWord>("ll_per_word")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getTermWeight>("tw")
		.property<&LDAModelObject::getDocs>("docs")
		.property<&LDAModelObject::getVocabs>("vocabs")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getV>("num_vocabs")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getVocabCf>("vocab_freq")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getVocabDf>("vocab_df")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getN>("num_words")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getOptimInterval, &tomoto::ILDAModel::setOptimInterval>("optim_interval")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getBurnInIteration, &tomoto::ILDAModel::setBurnInIteration>("burn_in")
		.property<&LDAModelObject::getRemovedTopWords>("removed_top_words")
		.property<&LDAModelObject::getUsedVocabs>("used_vocabs")
		.property<&LDAModelObject::getUsedVocabCf>("used_vocab_freq")
		.property<&LDAModelObject::getUsedVocabWeightedCf>("used_vocab_weighted_freq")
		.property<&LDAModelObject::getUsedVocabDf>("used_vocab_df")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getGlobalStep>("global_step"),

		py::define<DMRModelObject, LDAModelObject>("tomotopy.DMRModel", "_DMRModel", Py_TPFLAGS_BASETYPE)
		.method<&DMRModelObject::addDoc>("add_doc")
		.method<&DMRModelObject::makeDoc>("make_doc")
		.method<&DMRModelObject::getTopicPrior>("get_topic_prior")
		.property<&DMRModelObject::getMetadataDict>("metadata_dict")
		.property<&DMRModelObject::getMultiMetadataDict>("multi_metadata_dict")
		.property<&DMRModelObject::getLambda>("lambdas")
		.property<&DMRModelObject::getLambdaV2>("lambda_")
		.property<&DMRModelObject::getAlpha>("alpha")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getF>("f")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getSigma>("sigma")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getAlphaEps>("alpha_epsilon"),

		py::define<GDMRModelObject, DMRModelObject>("tomotopy.GDMRModel", "_GDMRModel", Py_TPFLAGS_BASETYPE)
		.method<&GDMRModelObject::addDoc>("add_doc")
		.method<&GDMRModelObject::makeDoc>("make_doc")
		.method<&GDMRModelObject::getTopicPrior>("get_topic_prior")
		.method<&GDMRModelObject::tdf>("tdf")
		.method<&GDMRModelObject::tdfLinspace>("tdf_linspace")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getFs>("degrees")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getSigma0>("sigma0")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getOrderDecay>("decay")
		.property<&GDMRModelObject::getMetadataRange>("metadata_range"),

		py::define<PAModelObject, LDAModelObject>("tomotopy.PAModel", "_PAModel", Py_TPFLAGS_BASETYPE)
		.method<&PAModelObject::getSubTopicPrior>("get_sub_topic_prior")
		.method<&PAModelObject::getSubTopics>("get_sub_topics")
		.method<&PAModelObject::getTopicWords>("get_topic_words")
		.method<&PAModelObject::getTopicWordDist>("get_topic_word_dist")
		.method<&PAModelObject::infer>("infer")
		.method<&PAModelObject::getCountBySuperTopic>("get_count_by_super_topic")
		.property<&PAModelObject::getSubAlpha>("subalpha")
		.property2<&PAModelObject::getInst<tomoto::IPAModel>, &tomoto::IPAModel::getK2>("k2"),

		py::define<HPAModelObject, PAModelObject>("tomotopy.HPAModel", "_HPAModel", Py_TPFLAGS_BASETYPE)
		.method<&HPAModelObject::getTopicWords>("get_topic_words")
		.method<&HPAModelObject::getTopicWordDist>("get_topic_word_dist")
		.method<&HPAModelObject::infer>("infer")
		.property<&HPAModelObject::getAlpha>("alpha")
		.property<&HPAModelObject::getSubAlpha>("subalpha"),

		py::define<MGLDAModelObject, LDAModelObject>("tomotopy.MGLDAModel", "_MGLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&MGLDAModelObject::addDoc>("add_doc")
		.method<&MGLDAModelObject::makeDoc>("make_doc")
		.method<&MGLDAModelObject::getTopicWords>("get_topic_words")
		.method<&MGLDAModelObject::getTopicWordDist>("get_topic_word_dist")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getKL>("k_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getT>("t")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaL>("alpha_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaM>("alpha_mg")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaML>("alpha_ml")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getEtaL>("eta_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getGamma>("gamma"),

		py::define<HDPModelObject, LDAModelObject>("tomotopy.HDPModel", "_HDPModel", Py_TPFLAGS_BASETYPE)
		.method<&HDPModelObject::isLiveTopic>("is_live_topic")
		.method<&HDPModelObject::convertToLDA>("convert_to_lda")
		.method<&HDPModelObject::purgeDeadTopics>("purge_dead_topics")
		.property<&HDPModelObject::getAlpha>("alpha")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getGamma>("gamma")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getLiveK>("live_k")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getTotalTables>("num_tables"),

		py::define<HLDAModelObject, LDAModelObject>("tomotopy.HLDAModel", "_HLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::isLiveTopic>>("is_live_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getNumDocsOfTopic>>("num_docs_of_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getLevelOfTopic>>("level")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getParentTopicId>>("parent_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getChildTopicId>>("children_topics")
		.property<&HLDAModelObject::getAlpha>("alpha")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getGamma>("gamma")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getLiveK>("live_k")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getLevelDepth>("depth"),

		py::define<CTModelObject, LDAModelObject>("tomotopy.CTModel", "_CTModel", Py_TPFLAGS_BASETYPE)
		.method<&CTModelObject::getCorrelations>("get_correlations")
		.property<&CTModelObject::getPriorCov>("prior_cov")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getPriorMean>("prior_mean")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getNumBetaSample, &tomoto::ICTModel::setNumBetaSample>("num_beta_samples")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getNumTMNSample, &tomoto::ICTModel::setNumTMNSample>("num_tmn_samples"),

		py::define<SLDAModelObject, LDAModelObject>("tomotopy.SLDAModel", "_SLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&SLDAModelObject::addDoc>("add_doc")
		.method<&SLDAModelObject::makeDoc>("make_doc")
		.method<&SLDAModelObject::getRegressionCoef>("get_regression_coef")
		.method<&SLDAModelObject::getTypeOfVar>("get_var_type")
		.method<&SLDAModelObject::estimateVars>("estimate")
		.property2<&SLDAModelObject::getInst<tomoto::ISLDAModel>, &tomoto::ISLDAModel::getF>("f"),

		py::define<LLDAModelObject, LDAModelObject>("tomotopy.LLDAModel", "_LLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&LLDAModelObject::addDoc>("add_doc")
		.method<&LLDAModelObject::makeDoc>("make_doc")
		.property<&LLDAModelObject::getTopicLabelDict>("topic_label_dict"),

		py::define<PLDAModelObject, LDAModelObject>("tomotopy.PLDA", "_PLDA", Py_TPFLAGS_BASETYPE)
		.method<&PLDAModelObject::addDoc>("add_doc")
		.method<&PLDAModelObject::makeDoc>("make_doc")
		.property<&PLDAModelObject::getTopicLabelDict>("topic_label_dict")
		.property2<&PLDAModelObject::getInst<tomoto::IPLDAModel>, &tomoto::IPLDAModel::getNumTopicsPerLabel>("topics_per_label")
		.property2<&PLDAModelObject::getInst<tomoto::IPLDAModel>, &tomoto::IPLDAModel::getNumLatentTopics>("latent_topics"),

		py::define<DTModelObject, LDAModelObject>("tomotopy.DTModel", "_DTModel", Py_TPFLAGS_BASETYPE)
		.method<&DTModelObject::addDoc>("add_doc")
		.method<&DTModelObject::makeDoc>("make_doc")
		.method<&DTModelObject::getAlpha2>("get_alpha")
		.method<&DTModelObject::getPhi>("get_phi")
		.method<&DTModelObject::getTopicWords>("get_topic_words")
		.method<&DTModelObject::getTopicWordDist>("get_topic_word_dist")
		.method<&DTModelObject::getCountByTopic>("get_count_by_topics")
		.property<&DTModelObject::getAlpha>("alpha")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeA, &tomoto::IDTModel::setShapeA>("lr_a")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeB, &tomoto::IDTModel::setShapeB>("lr_b")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeC, &tomoto::IDTModel::setShapeC>("lr_c")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getT>("num_timepoints")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getNumDocsByT>("num_docs_by_timepoint"),

		py::define<PTModelObject, LDAModelObject>("tomotopy.PTModel", "_PTModel", Py_TPFLAGS_BASETYPE)
		.property2<&PTModelObject::getInst<tomoto::IPTModel>, &tomoto::IPTModel::getP>("p")
	);

#ifdef __AVX2__
	PyModule_AddStringConstant(moduleObj, "isa", "avx2");
#elif defined(__AVX__)
	PyModule_AddStringConstant(moduleObj, "isa", "avx");
#elif defined(__SSE2__) || defined(__x86_64__) || defined(_WIN64)
	PyModule_AddStringConstant(moduleObj, "isa", "sse2");
#else
	PyModule_AddStringConstant(moduleObj, "isa", isa_str);
#endif
	addLabelTypes(module);
	addUtilsTypes(module);
	addCoherenceTypes(module);

	return moduleObj;
}
