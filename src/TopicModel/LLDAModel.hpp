#pragma once
#include "LDAModel.hpp"
#include "LLDA.h"

/*
Implementation of Labeled LDA using Gibbs sampling by bab2min

* Ramage, D., Hall, D., Nallapati, R., & Manning, C. D. (2009, August). Labeled LDA: A supervised topic model for credit attribution in multi-labeled corpora. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing: Volume 1-Volume 1 (pp. 248-256). Association for Computational Linguistics.
*/

namespace tomoto
{
	template<TermWeight _tw,
		typename _Interface = ILLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentLLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>>
	class LLDAModel : public LDAModel<_tw, flags::generator_by_doc | flags::partitioned_multisampling, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, LLDAModel<_tw>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, LLDAModel<_tw>, _Derived>::type;
		using BaseClass = LDAModel<_tw, flags::generator_by_doc | flags::partitioned_multisampling, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr char TMID[] = "LLDA";

		Dictionary topicLabelDict;

		template<bool _asymEta>
		Float* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<Float>() + this->alphas.array())
				* (ld.numByTopicWord.col(vid).array().template cast<Float>() + this->eta)
				/ (ld.numByTopic.array().template cast<Float>() + V * this->eta);
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

		struct Generator
		{
			std::discrete_distribution<> theta;
		};

		Generator makeGeneratorForInit(const _DocType* doc) const
		{
			std::discrete_distribution<> theta{ doc->labelMask.data(), doc->labelMask.data() + this->K };
			return Generator{ theta };
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
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 0, topicLabelDict);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseClass, 1, 0x00010001, topicLabelDict);

		LLDAModel(size_t _K = 1, Float _alpha = 1.0, Float _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_K, _alpha, _eta, _rg)
		{
		}

		template<bool _const = false>
		_DocType& _updateDoc(_DocType& doc, const std::vector<std::string>& labels)
		{
			if (_const)
			{
				doc.labelMask.resize(this->K);
				doc.labelMask.setOnes();

				std::vector<Vid> topicLabelIds;
				for (auto& label : labels)
				{
					auto tid = topicLabelDict.toWid(label);
					if (tid == (Vid)-1) continue;
					topicLabelIds.emplace_back(tid);
				}

				if (!topicLabelIds.empty())
				{
					doc.labelMask.head(topicLabelDict.size()).setZero();
					for (auto tid : topicLabelIds) doc.labelMask[tid] = 1;
				}
			}
			else
			{
				if (!labels.empty())
				{
					std::vector<Vid> topicLabelIds;
					for (auto& label : labels) topicLabelIds.emplace_back(topicLabelDict.add(label));
					auto maxVal = *std::max_element(topicLabelIds.begin(), topicLabelIds.end());
					doc.labelMask.resize(maxVal + 1);
					doc.labelMask.setZero();
					for (auto i : topicLabelIds) doc.labelMask[i] = 1;
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

		size_t getNumTopicsPerLabel() const override { return 1; }
	};
}
