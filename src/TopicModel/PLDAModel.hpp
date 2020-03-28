#pragma once
#include "LDAModel.hpp"
#include "PLDA.h"

/*
Implementation of Labeled LDA using Gibbs sampling by bab2min

* Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.
*/

namespace tomoto
{
	template<TermWeight _tw,
		typename _Interface = IPLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentLLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>>
	class PLDAModel : public LDAModel<_tw, flags::generator_by_doc | flags::partitioned_multisampling, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_tw>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, flags::generator_by_doc | flags::partitioned_multisampling, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr char TMID[] = "PLDA";

		Dictionary topicLabelDict;

		size_t numLatentTopics, numTopicsPerLabel;

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto etaHelper = this->template getEtaHelper<_asymEta>();
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + this->alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + etaHelper.getEta(vid))
				/ (ld.numByTopic.array().template cast<Float>() + etaHelper.getEtaSum());
			zLikelihood.array() *= doc.labelMask.array().template cast<Float>();
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

		struct Generator
		{
			std::discrete_distribution<> theta;
		};

		Generator makeGeneratorForInit(const _DocType* doc) const
		{
			return Generator{ 
				std::discrete_distribution<>{ doc->labelMask.data(), doc->labelMask.data() + doc->labelMask.size() } 
			};
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			if (this->etaByTopicWord.size())
			{
				Eigen::Array<Float, -1, 1> col = this->etaByTopicWord.col(w);
				for (size_t k = 0; k < col.size(); ++k) col[k] *= g.theta.probabilities()[k];
				z = sample::sampleFromDiscrete(col.data(), col.data() + col.size(), rgs);
			}
			else
			{
				z = g.theta(rgs);
			}
			this->template addWordTo<1>(ld, doc, i, w, z);
		}

	public:
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, topicLabelDict, numLatentTopics, numTopicsPerLabel);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, topicLabelDict, numLatentTopics, numTopicsPerLabel);

		PLDAModel(size_t _numLatentTopics = 0, size_t _numTopicsPerLabel = 1, 
			Float _alpha = 1.0, Float _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(1, _alpha, _eta, _rg), 
			numLatentTopics(_numLatentTopics), numTopicsPerLabel(_numTopicsPerLabel)
		{
			if (_numLatentTopics >= 0x80000000) 
				THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong numLatentTopics value (numLatentTopics = %zd)", _numLatentTopics));
			if (_numTopicsPerLabel == 0 || _numTopicsPerLabel >= 0x80000000) 
				THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong numTopicsPerLabel value (numTopicsPerLabel = %zd)", _numTopicsPerLabel));
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<std::string>& labels)
		{
			if (_const)
			{
				doc.labelMask.resize(this->K);
				doc.labelMask.setZero();
				doc.labelMask.tail(numLatentTopics).setOnes();

				std::vector<Vid> topicLabelIds;
				for (auto& label : labels)
				{
					auto tid = topicLabelDict.toWid(label);
					if (tid == (Vid)-1) continue;
					topicLabelIds.emplace_back(tid);
				}

				for (auto tid : topicLabelIds) doc.labelMask.segment(tid * numTopicsPerLabel, numTopicsPerLabel).setOnes();
				if (labels.empty()) doc.labelMask.setOnes();
			}
			else
			{
				if (!labels.empty())
				{
					std::vector<Vid> topicLabelIds;
					for (auto& label : labels) topicLabelIds.emplace_back(topicLabelDict.add(label));
					auto maxVal = *std::max_element(topicLabelIds.begin(), topicLabelIds.end());
					doc.labelMask.resize((maxVal + 1) * numTopicsPerLabel);
					doc.labelMask.setZero();
					for (auto i : topicLabelIds) doc.labelMask.segment(i * numTopicsPerLabel, numTopicsPerLabel).setOnes();
				}
			}
			return doc;
		}

		size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) override
		{
			auto doc = this->_makeDoc(words);
			return this->_addDoc(_updateDoc(doc, labels));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& labels) const override
		{
			auto doc = as_mutable(this)->template _makeDoc<true>(words);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, labels));
		}

		size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<std::string>& labels) override
		{
			auto doc = this->template _makeRawDoc<false>(rawStr, tokenizer);
			return this->_addDoc(_updateDoc(doc, labels));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<std::string>& labels) const override
		{
			auto doc = as_mutable(this)->template _makeRawDoc<true>(rawStr, tokenizer);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, labels));
		}

		size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<std::string>& labels) override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return this->_addDoc(_updateDoc(doc, labels));
		}

		std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<std::string>& labels) const override
		{
			auto doc = this->_makeRawDoc(rawStr, words, pos, len);
			return make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, labels));
		}

		std::vector<Float> getTopicsByDoc(const _DocType& doc) const
		{
			std::vector<Float> ret(this->K);
			auto maskedAlphas = this->alphas.array() * doc.labelMask.template cast<Float>().array();
			Eigen::Map<Eigen::Matrix<Float, -1, 1>> { ret.data(), this->K }.array() =
				(doc.numByTopic.array().template cast<Float>() + maskedAlphas)
				/ (doc.getSumWordWeight() + maskedAlphas.sum());
			return ret;
		}

		const Dictionary& getTopicLabelDict() const override { return topicLabelDict; }

		size_t getNumLatentTopics() const override { return numLatentTopics; }

		size_t getNumTopicsPerLabel() const override { return numTopicsPerLabel; }
	};
}
