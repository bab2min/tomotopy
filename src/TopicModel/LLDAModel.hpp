#pragma once
#include "LDAModel.hpp"
#include "LLDA.h"

/*
Implementation of Labeled LDA using Gibbs sampling by bab2min

* Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.
*/

namespace tomoto
{
	template<TermWeight _TW,
		typename _Interface = ILLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentLLDA<_TW>,
		typename _ModelState = ModelStateLDA<_TW>>
	class LLDAModel : public LDAModel<_TW, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LLDAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "LLDA";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LLDAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Dictionary topicLabelDict;

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + this->alphas.array())
				* (doc.labelMask.array().template cast<FLOAT>())
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);

			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
			if (doc.labelMask.size() == 0)
			{
				doc.labelMask.resize(this->K);
				doc.labelMask.setOnes();
			}
			else if (doc.labelMask.size() < this->K)
			{
				size_t oldSize = doc.labelMask.size();
				doc.labelMask.conservativeResize(this->K);
				doc.labelMask.segment(oldSize, topicLabelDict.size() - oldSize).setZero();
				doc.labelMask.segment(topicLabelDict.size(), this->K - topicLabelDict.size()).setOnes();
			}
		}

		void initGlobalState(bool initDocs)
		{
			this->K = std::max(topicLabelDict.size(), (size_t)this->K);
			this->alphas.resize(this->K);
			this->alphas.array() = this->alpha;
			BaseClass::initGlobalState(initDocs);
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, topicLabelDict);

	public:
		LLDAModel(size_t _K = 1, FLOAT _alpha = 1.0, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, _alpha, _eta, _rg)
		{
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) override
		{	
			auto doc = this->_makeDoc(words);

			if (!labels.empty())
			{
				std::vector<VID> topicLabelIds;
				for (auto& label : labels) topicLabelIds.emplace_back(topicLabelDict.add(label));
				auto maxVal = *std::max_element(topicLabelIds.begin(), topicLabelIds.end());
				doc.labelMask.resize(maxVal + 1);
				doc.labelMask.setZero();
				for (auto i : topicLabelIds) doc.labelMask[i] = 1;
			}
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) const override
		{
			auto doc = this->_makeDocWithinVocab(words);
			doc.labelMask.resize(this->K);
			doc.labelMask.setOnes();

			std::vector<VID> topicLabelIds;
			for (auto& label : labels)
			{
				auto tid = topicLabelDict.toWid(label);
				if (tid == (VID)-1) continue;
				topicLabelIds.emplace_back(tid);
			}

			if (!topicLabelIds.empty())
			{
				doc.labelMask.head(topicLabelDict.size()).setZero();
				for (auto tid : topicLabelIds) doc.labelMask[tid] = 1;
			}

			return make_unique<_DocType>(doc);
		}

		const Dictionary& getTopicLabelDict() const { return topicLabelDict; }
	};
}
