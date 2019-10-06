#pragma once
#include "TopicModel.hpp"

namespace tomoto
{
    enum class TermWeight { one, idf, pmi, size };

	template<typename _Scalar>
	struct ShareableVector : Eigen::Map<Eigen::Matrix<_Scalar, -1, 1>>
	{
		Eigen::Matrix<_Scalar, -1, 1> ownData;
		ShareableVector(_Scalar* ptr = nullptr, Eigen::Index len = 0) 
			: Eigen::Map<Eigen::Matrix<_Scalar, -1, 1>>(nullptr, 0)
		{
			init(ptr, len);
		}

		void init(_Scalar* ptr, Eigen::Index len)
		{
			if (!ptr && len)
			{
				ownData = Eigen::Matrix<_Scalar, -1, 1>::Zero(len);
				ptr = ownData.data();
			}
			// is this the best way??
			this->m_data = ptr;
			((Eigen::internal::variable_if_dynamic<Eigen::Index, -1>*)&this->m_rows)->setValue(len);
		}

		void conservativeResize(size_t newSize)
		{
			ownData.conservativeResize(newSize);
			init(ownData.data(), ownData.size());
		}

		void becomeOwner()
		{
			if (ownData.data() != this->m_data)
			{
				ownData = *this;
				init(ownData.data(), ownData.size());
			}
		}
	};

	template<TermWeight _TW>
	struct SumWordWeight
	{
		FLOAT sumWordWeight = 0;
	};

	template<>
	struct SumWordWeight<TermWeight::one>
	{
	};

	template<TermWeight _TW, bool _Shared = false>
	struct DocumentLDA : public DocumentBase, SumWordWeight<_TW>
	{
	public:
		using DocumentBase::DocumentBase;
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		tvector<TID> Zs;
		tvector<FLOAT> wordWeights;
		ShareableVector<WeightType> numByTopic;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentBase, Zs, wordWeights);

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);

		template<TermWeight __TW>
		typename std::enable_if<__TW == TermWeight::one, int32_t>::type getSumWordWeight() const
		{
			return this->words.size();
		}

		template<TermWeight __TW>
		typename std::enable_if<__TW != TermWeight::one, FLOAT>::type getSumWordWeight() const
		{
			//return std::accumulate(wordWeights.begin(), wordWeights.end(), 0.f);
			return this->sumWordWeight;
		}

		template<TermWeight __TW>
		typename std::enable_if<__TW == TermWeight::one>::type updateSumWordWeight()
		{
		}

		template<TermWeight __TW>
		typename std::enable_if<__TW != TermWeight::one>::type updateSumWordWeight()
		{
			this->sumWordWeight = std::accumulate(wordWeights.begin(), wordWeights.end(), 0.f);
		}
	};

    class ILDAModel : public ITopicModel
	{
	public:
		using DefaultDocType = DocumentLDA<TermWeight::one>;
		static ILDAModel* create(TermWeight _weight, size_t _K = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words) const = 0;

		virtual TermWeight getTermWeight() const = 0;
		virtual size_t getOptimInterval() const = 0;
		virtual void setOptimInterval(size_t) = 0;
		virtual size_t getBurnInIteration() const = 0;
		virtual void setBurnInIteration(size_t) = 0;
		virtual std::vector<size_t> getCountByTopic() const = 0;
		virtual size_t getK() const = 0;
		virtual FLOAT getAlpha() const = 0;
		virtual FLOAT getAlpha(TID k1) const = 0;
		virtual FLOAT getEta() const = 0;
	};
}