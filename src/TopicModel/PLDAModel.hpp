#pragma once
#include "LDAModel.hpp"
#include "PLDA.h"

/*
Implementation of Labeled LDA using Gibbs sampling by bab2min

* Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.
*/

namespace tomoto
{
	template<TermWeight _TW,
		typename _Interface = IPLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentLLDA<_TW>,
		typename _ModelState = ModelStateLDA<_TW>>
	class PLDAModel : public LDAModel<_TW, 0, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "PLDA";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, 0, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		Dictionary topicLabelDict;

		size_t numLatentTopics, numTopicsPerLabel;

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + this->alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<FLOAT>() + this->eta)
				/ (ld.numByTopic.array().template cast<FLOAT>() + V * this->eta);
			zLikelihood.array() *= doc.labelMask.array().template cast<FLOAT>();
			sample::prefixSum(zLikelihood.data(), this->K);
			return &zLikelihood[0];
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, topicDocPtr, wordSize);
			if (doc.labelMask.size() == 0)
			{
				doc.labelMask.resize(this->K);
				doc.labelMask.setZero();
				doc.labelMask.tail(numLatentTopics).setOnes();
			}
			else if (doc.labelMask.size() < this->K)
			{
				size_t oldSize = doc.labelMask.size();
				doc.labelMask.conservativeResize(this->K);
				doc.labelMask.tail(this->K - oldSize).setZero();
				doc.labelMask.tail(numLatentTopics).setOnes();
			}
		}

		void initGlobalState(bool initDocs)
		{
			this->K = topicLabelDict.size() * numTopicsPerLabel + numLatentTopics;
			this->alphas.resize(this->K);
			this->alphas.array() = this->alpha;
			BaseClass::initGlobalState(initDocs);
		}

		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, topicLabelDict, numLatentTopics, numTopicsPerLabel);

	public:
		PLDAModel(size_t _numLatentTopics = 0, size_t _numTopicsPerLabel = 1, 
			FLOAT _alpha = 1.0, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(1, _alpha, _eta, _rg), 
			numLatentTopics(_numLatentTopics), numTopicsPerLabel(_numTopicsPerLabel)
		{
			if (_numLatentTopics >= 0x80000000) 
				THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong numLatentTopics value (numLatentTopics = %zd)", _numLatentTopics));
			if (_numTopicsPerLabel == 0 || _numTopicsPerLabel >= 0x80000000) 
				THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong numTopicsPerLabel value (numTopicsPerLabel = %zd)", _numTopicsPerLabel));
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) override
		{
			auto doc = this->_makeDoc(words);

			if (!labels.empty())
			{
				std::vector<VID> topicLabelIds;
				for (auto& label : labels) topicLabelIds.emplace_back(topicLabelDict.add(label));
				auto maxVal = *std::max_element(topicLabelIds.begin(), topicLabelIds.end());
				doc.labelMask.resize((maxVal + 1) * numTopicsPerLabel);
				doc.labelMask.setZero();
				for (auto i : topicLabelIds) doc.labelMask.segment(i * numTopicsPerLabel, numTopicsPerLabel).setOnes();
			}
			return this->_addDoc(doc);
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) const override
		{
			auto doc = this->_makeDocWithinVocab(words);
			doc.labelMask.resize(this->K);
			doc.labelMask.setZero();
			doc.labelMask.tail(numLatentTopics).setOnes();

			std::vector<VID> topicLabelIds;
			for (auto& label : labels)
			{
				auto tid = topicLabelDict.toWid(label);
				if (tid == (VID)-1) continue;
				topicLabelIds.emplace_back(tid);
			}

			for (auto tid : topicLabelIds) doc.labelMask.segment(tid * numTopicsPerLabel, numTopicsPerLabel).setOnes();

			return make_unique<_DocType>(doc);
		}

		const Dictionary& getTopicLabelDict() const { return topicLabelDict; }

		size_t getNumLatentTopics() const { return numLatentTopics; }

		size_t getNumTopicsPerLabel() const { return numTopicsPerLabel; }
	};
}
