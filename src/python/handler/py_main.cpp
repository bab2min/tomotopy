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

py::Module module{ "tomotopy", "Tomoto Module for Python" };

PyMODINIT_FUNC MODULE_NAME()
{
	auto moduleObj = module.init(
		py::defineP<LDAModelObject>("tomotopy._LDAModel", "_LDAModel", Py_TPFLAGS_BASETYPE)
		.method<&LDAModelObject::addDoc>("_add_doc")
		.method<&LDAModelObject::addCorpus>("_add_corpus")
		.method<&LDAModelObject::makeDoc>("_make_doc")
		.method<&LDAModelObject::setWordPrior>("_set_word_prior")
		.method<&LDAModelObject::getWordPrior>("_get_word_prior")
		.method<&LDAModelObject::train>("_train")
		.method<&LDAModelObject::getCountByTopics>("_get_count_by_topics")
		.method<&LDAModelObject::getTopicWords>("_get_topic_words")
		.method<&LDAModelObject::getTopicWordDist>("_get_topic_word_dist")
		.method<&LDAModelObject::infer>("_infer")
		.method<&LDAModelObject::save>("_save")
		.method<&LDAModelObject::saves>("_saves")
		.staticMethod<&LDAModelObject::load>("_load")
		.staticMethod<&LDAModelObject::loads>("_loads")
		.method<&LDAModelObject::copy>("_copy")
		.method<&LDAModelObject::updateVocab>("__update_vocab")
		.method<&LDAModelObject::getWordForms>("_get_word_forms")
		.method<&LDAModelObject::getHash>("_get_hash")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getK>("_k")
		.property<&LDAModelObject::getAlpha>("_alpha")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getEta>("_eta")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getPerplexity>("_perplexity")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getLLPerWord>("_ll_per_word")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getTermWeight>("_tw")
		.property<&LDAModelObject::getDocs>("_docs")
		.property<&LDAModelObject::getVocabs>("_vocabs")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getV>("_num_vocabs")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getVocabCf>("_vocab_freq")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getVocabDf>("_vocab_df")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getN>("_num_words")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getOptimInterval, &tomoto::ILDAModel::setOptimInterval>("_optim_interval")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getBurnInIteration, &tomoto::ILDAModel::setBurnInIteration>("_burn_in")
		.property<&LDAModelObject::getRemovedTopWords>("_removed_top_words")
		.property<&LDAModelObject::getUsedVocabs>("_used_vocabs")
		.property<&LDAModelObject::getUsedVocabCf>("_used_vocab_freq")
		.property<&LDAModelObject::getUsedVocabWeightedCf>("_used_vocab_weighted_freq")
		.property<&LDAModelObject::getUsedVocabDf>("_used_vocab_df")
		.property2<&LDAModelObject::getInst<tomoto::ILDAModel>, &tomoto::ILDAModel::getGlobalStep>("_global_step"),

		py::defineP<DMRModelObject, LDAModelObject>("tomotopy._DMRModel", "_DMRModel", Py_TPFLAGS_BASETYPE)
		.method<&DMRModelObject::addDoc>("_add_doc")
		.method<&DMRModelObject::makeDoc>("_make_doc")
		.method<&DMRModelObject::getTopicPrior>("_get_topic_prior")
		.property<&DMRModelObject::getMetadataDict>("_metadata_dict")
		.property<&DMRModelObject::getMultiMetadataDict>("_multi_metadata_dict")
		.property<&DMRModelObject::getLambda>("_lambdas")
		.property<&DMRModelObject::getLambdaV2>("_lambda_")
		.property<&DMRModelObject::getAlpha>("_alpha")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getF>("_f")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getSigma>("_sigma")
		.property2<&DMRModelObject::getInst<tomoto::IDMRModel>, &tomoto::IDMRModel::getAlphaEps>("_alpha_epsilon"),

		py::defineP<GDMRModelObject, DMRModelObject>("tomotopy._GDMRModel", "_GDMRModel", Py_TPFLAGS_BASETYPE)
		.method<&GDMRModelObject::addDoc>("_add_doc")
		.method<&GDMRModelObject::makeDoc>("_make_doc")
		.method<&GDMRModelObject::getTopicPrior>("_get_topic_prior")
		.method<&GDMRModelObject::tdf>("_tdf")
		.method<&GDMRModelObject::tdfLinspace>("_tdf_linspace")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getFs>("_degrees")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getSigma0>("_sigma0")
		.property2<&GDMRModelObject::getInst<tomoto::IGDMRModel>, &tomoto::IGDMRModel::getOrderDecay>("_decay")
		.property<&GDMRModelObject::getMetadataRange>("_metadata_range"),

		py::defineP<PAModelObject, LDAModelObject>("tomotopy._PAModel", "_PAModel", Py_TPFLAGS_BASETYPE)
		.method<&PAModelObject::getSubTopicPrior>("_get_sub_topic_prior")
		.method<&PAModelObject::getSubTopics>("_get_sub_topics")
		.method<&PAModelObject::getTopicWords>("_get_topic_words")
		.method<&PAModelObject::getTopicWordDist>("_get_topic_word_dist")
		.method<&PAModelObject::infer>("_infer")
		.method<&PAModelObject::getCountBySuperTopic>("_get_count_by_super_topic")
		.property<&PAModelObject::getSubAlpha>("_subalpha")
		.property2<&PAModelObject::getInst<tomoto::IPAModel>, &tomoto::IPAModel::getK2>("_k2"),

		py::defineP<HPAModelObject, PAModelObject>("tomotopy._HPAModel", "_HPAModel", Py_TPFLAGS_BASETYPE)
		.method<&HPAModelObject::getTopicWords>("_get_topic_words")
		.method<&HPAModelObject::getTopicWordDist>("_get_topic_word_dist")
		.method<&HPAModelObject::infer>("_infer")
		.property<&HPAModelObject::getAlpha>("_alpha")
		.property<&HPAModelObject::getSubAlpha>("_subalpha"),

		py::defineP<MGLDAModelObject, LDAModelObject>("tomotopy._MGLDAModel", "_MGLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&MGLDAModelObject::addDoc>("_add_doc")
		.method<&MGLDAModelObject::makeDoc>("_make_doc")
		.method<&MGLDAModelObject::getTopicWords>("_get_topic_words")
		.method<&MGLDAModelObject::getTopicWordDist>("_get_topic_word_dist")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getKL>("_k_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getT>("_t")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaL>("_alpha_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaM>("_alpha_mg")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getAlphaML>("_alpha_ml")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getEtaL>("_eta_l")
		.property2<&MGLDAModelObject::getInst<tomoto::IMGLDAModel>, &tomoto::IMGLDAModel::getGamma>("_gamma"),

		py::defineP<HDPModelObject, LDAModelObject>("tomotopy._HDPModel", "_HDPModel", Py_TPFLAGS_BASETYPE)
		.method<&HDPModelObject::isLiveTopic>("_is_live_topic")
		.method<&HDPModelObject::convertToLDA>("_convert_to_lda")
		.method<&HDPModelObject::purgeDeadTopics>("_purge_dead_topics")
		.property<&HDPModelObject::getAlpha>("_alpha")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getGamma>("_gamma")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getLiveK>("_live_k")
		.property2<&HDPModelObject::getInst<tomoto::IHDPModel>, &tomoto::IHDPModel::getTotalTables>("_num_tables"),

		py::defineP<HLDAModelObject, LDAModelObject>("tomotopy._HLDAModel", "_HLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::isLiveTopic>>("_is_live_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getNumDocsOfTopic>>("_num_docs_of_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getLevelOfTopic>>("_level")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getParentTopicId>>("_parent_topic")
		.method<&HLDAModelObject::getTopicInfo<&tomoto::IHLDAModel::getChildTopicId>>("_children_topics")
		.property<&HLDAModelObject::getAlpha>("_alpha")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getGamma>("_gamma")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getLiveK>("_live_k")
		.property2<&HLDAModelObject::getInst<tomoto::IHLDAModel>, &tomoto::IHLDAModel::getLevelDepth>("_depth"),

		py::defineP<CTModelObject, LDAModelObject>("tomotopy._CTModel", "_CTModel", Py_TPFLAGS_BASETYPE)
		.method<&CTModelObject::getCorrelations>("_get_correlations")
		.property<&CTModelObject::getPriorCov>("_prior_cov")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getPriorMean>("_prior_mean")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getNumBetaSample, &tomoto::ICTModel::setNumBetaSample>("_num_beta_samples")
		.property2<&CTModelObject::getInst<tomoto::ICTModel>, &tomoto::ICTModel::getNumTMNSample, &tomoto::ICTModel::setNumTMNSample>("_num_tmn_samples"),

		py::defineP<SLDAModelObject, LDAModelObject>("tomotopy._SLDAModel", "_SLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&SLDAModelObject::addDoc>("_add_doc")
		.method<&SLDAModelObject::makeDoc>("_make_doc")
		.method<&SLDAModelObject::getRegressionCoef>("_get_regression_coef")
		.method<&SLDAModelObject::getTypeOfVar>("_get_var_type")
		.method<&SLDAModelObject::estimateVars>("_estimate")
		.property2<&SLDAModelObject::getInst<tomoto::ISLDAModel>, &tomoto::ISLDAModel::getF>("_f"),

		py::defineP<LLDAModelObject, LDAModelObject>("tomotopy._LLDAModel", "_LLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&LLDAModelObject::addDoc>("_add_doc")
		.method<&LLDAModelObject::makeDoc>("_make_doc")
		.property<&LLDAModelObject::getTopicLabelDict>("_topic_label_dict"),

		py::defineP<PLDAModelObject, LDAModelObject>("tomotopy._PLDAModel", "_PLDAModel", Py_TPFLAGS_BASETYPE)
		.method<&PLDAModelObject::addDoc>("_add_doc")
		.method<&PLDAModelObject::makeDoc>("_make_doc")
		.property<&PLDAModelObject::getTopicLabelDict>("_topic_label_dict")
		.property2<&PLDAModelObject::getInst<tomoto::IPLDAModel>, &tomoto::IPLDAModel::getNumTopicsPerLabel>("_topics_per_label")
		.property2<&PLDAModelObject::getInst<tomoto::IPLDAModel>, &tomoto::IPLDAModel::getNumLatentTopics>("_latent_topics"),

		py::defineP<DTModelObject, LDAModelObject>("tomotopy._DTModel", "_DTModel", Py_TPFLAGS_BASETYPE)
		.method<&DTModelObject::addDoc>("_add_doc")
		.method<&DTModelObject::makeDoc>("_make_doc")
		.method<&DTModelObject::getAlpha2>("_get_alpha")
		.method<&DTModelObject::getPhi>("_get_phi")
		.method<&DTModelObject::getTopicWords>("_get_topic_words")
		.method<&DTModelObject::getTopicWordDist>("_get_topic_word_dist")
		.method<&DTModelObject::getCountByTopic>("_get_count_by_topics")
		.property<&DTModelObject::getAlpha>("_alpha")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeA, &tomoto::IDTModel::setShapeA>("_lr_a")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeB, &tomoto::IDTModel::setShapeB>("_lr_b")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getShapeC, &tomoto::IDTModel::setShapeC>("_lr_c")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getT>("_num_timepoints")
		.property2<&DTModelObject::getInst<tomoto::IDTModel>, &tomoto::IDTModel::getNumDocsByT>("_num_docs_by_timepoint"),

		py::defineP<PTModelObject, LDAModelObject>("tomotopy._PTModel", "_PTModel", Py_TPFLAGS_BASETYPE)
		.property2<&PTModelObject::getInst<tomoto::IPTModel>, &tomoto::IPTModel::getP>("_p")
	);

#ifdef __AVX512F__
	PyModule_AddStringConstant(moduleObj, "isa", "avx512");
#elif defined(__AVX2__)
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
