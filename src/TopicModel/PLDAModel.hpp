#pragma once
#include "LDAModel.hpp"
#include "PLDA.h"

/*
Implementation of Labeled LDA using Gibbs sampling by bab2min

* Ramage, D., Manning, C. D., & Dumais, S. (2011, August). Partially labeled topic models for interpretable text mining. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 457-465). ACM.
*/

namespace tomoto
{
	template<TermWeight _tw, typename _RandGen,
		typename _Interface = IPLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentLLDA<_tw>,
		typename _ModelState = ModelStateLDA<_tw>>
	class PLDAModel : public LDAModel<_tw, _RandGen, flags::generator_by_doc | flags::partitioned_multisampling, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_tw, _RandGen>, _Derived>::type,
		_DocType, _ModelState>
	{
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, PLDAModel<_tw, _RandGen>, _Derived>::type;
		using BaseClass = LDAModel<_tw, _RandGen, flags::generator_by_doc | flags::partitioned_multisampling, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		static constexpr auto tmid()
		{
			return serializer::to_key("PLDA");
		}

		Dictionary topicLabelDict;

		uint64_t numLatentTopics, numTopicsPerLabel;

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

		void prepareDoc(_DocType& doc, size_t docId, size_t wordSize) const
		{
			BaseClass::prepareDoc(doc, docId, wordSize);
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
			Eigen::Array<Float, -1, 1> p;
			Eigen::Rand::DiscreteGen<int32_t> theta;
		};

		Generator makeGeneratorForInit(const _DocType* doc) const
		{
			Generator g;
			g.p = doc->labelMask.array().template cast<Float>() * this->alphas.array();
			g.theta = Eigen::Rand::DiscreteGen<int32_t>{ g.p.data(), g.p.data() + this->K };
			return g;
		}

		template<bool _Infer>
		void updateStateWithDoc(Generator& g, _ModelState& ld, _RandGen& rgs, _DocType& doc, size_t i) const
		{
			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			if (this->etaByTopicWord.size())
			{
				Eigen::Array<Float, -1, 1> col = this->etaByTopicWord.col(w);
				col *= g.p;
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

		PLDAModel(const PLDAArgs& args)
			: BaseClass(args.setK(1)),
			numLatentTopics(args.numLatentTopics), numTopicsPerLabel(args.numTopicsPerLabel)
		{
			if (numLatentTopics >= 0x80000000)
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong numLatentTopics value (numLatentTopics = %zd)", numLatentTopics));
			if (numTopicsPerLabel == 0 || numTopicsPerLabel >= 0x80000000)
				THROW_ERROR_WITH_INFO(exc::InvalidArgument, text::format("wrong numTopicsPerLabel value (numTopicsPerLabel = %zd)", numTopicsPerLabel));
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

		size_t addDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) override
		{
			auto doc = this->template _makeFromRawDoc<false>(rawDoc, tokenizer);
			return this->_addDoc(_updateDoc(doc, rawDoc.template getMiscDefault<std::vector<std::string>>("labels")));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc, const RawDocTokenizer::Factory& tokenizer) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc, tokenizer);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, rawDoc.template getMiscDefault<std::vector<std::string>>("labels")));
		}

		size_t addDoc(const RawDoc& rawDoc) override
		{
			auto doc = this->_makeFromRawDoc(rawDoc);
			return this->_addDoc(_updateDoc(doc, rawDoc.template getMiscDefault<std::vector<std::string>>("labels")));
		}

		std::unique_ptr<DocumentBase> makeDoc(const RawDoc& rawDoc) const override
		{
			auto doc = as_mutable(this)->template _makeFromRawDoc<true>(rawDoc);
			return std::make_unique<_DocType>(as_mutable(this)->template _updateDoc<true>(doc, rawDoc.template getMiscDefault<std::vector<std::string>>("labels")));
		}

		std::vector<Float> _getTopicsByDoc(const _DocType& doc, bool normalize) const
		{
			if (!doc.numByTopic.size()) return {};
			std::vector<Float> ret(this->K);
			auto maskedAlphas = this->alphas.array() * doc.labelMask.template cast<Float>().array();
			Eigen::Map<Eigen::Array<Float, -1, 1>> m{ ret.data(), this->K };
			if (normalize)
			{
				m = (doc.numByTopic.array().template cast<Float>() + maskedAlphas)
					/ (doc.getSumWordWeight() + maskedAlphas.sum());
			}
			else
			{
				m = doc.numByTopic.array().template cast<Float>() + maskedAlphas;
			}
			return ret;
		}

		const Dictionary& getTopicLabelDict() const override { return topicLabelDict; }

		size_t getNumLatentTopics() const override { return numLatentTopics; }

		size_t getNumTopicsPerLabel() const override { return numTopicsPerLabel; }
	};
}
