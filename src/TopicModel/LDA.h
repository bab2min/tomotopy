#pragma once
#include "TopicModel.hpp"

namespace tomoto
{
    enum class TermWeight { one, idf, pmi, idf_one, size };

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

	template<typename _Base, TermWeight _TW>
	struct SumWordWeight
	{
		FLOAT sumWordWeight = 0;
		FLOAT getSumWordWeight() const
		{
			return sumWordWeight;
		}

		void updateSumWordWeight(size_t realV)
		{
			sumWordWeight = std::accumulate(static_cast<_Base*>(this)->wordWeights.begin(), static_cast<_Base*>(this)->wordWeights.end(), 0.f);
		}
	};

	template<typename _Base>
	struct SumWordWeight<_Base, TermWeight::one>
	{
		int32_t sumWordWeight = 0;
		int32_t getSumWordWeight() const
		{
			return sumWordWeight;
		}

		void updateSumWordWeight(size_t realV)
		{
			sumWordWeight = std::count_if(static_cast<_Base*>(this)->words.begin(), static_cast<_Base*>(this)->words.end(), [realV](VID w)
			{
				return w < realV;
			});
		}
	};

	template<TermWeight _TW, size_t _Flags = 0>
	struct DocumentLDA : public DocumentBase, SumWordWeight<DocumentLDA<_TW, _Flags>, _TW>
	{
	public:
		using DocumentBase::DocumentBase;
		using WeightType = typename std::conditional<_TW == TermWeight::one, int32_t, float>::type;

		tvector<TID> Zs;
		tvector<FLOAT> wordWeights;
		ShareableVector<WeightType> numByTopic;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentBase, Zs, wordWeights);

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
		
		WeightType getWordWeight(size_t idx) const
		{
			return _TW == TermWeight::one ? 1 : wordWeights[idx];
		}
	};

    class ILDAModel : public ITopicModel
	{
	public:
		using DefaultDocType = DocumentLDA<TermWeight::one>;
		static ILDAModel* create(TermWeight _weight, size_t _K = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });

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