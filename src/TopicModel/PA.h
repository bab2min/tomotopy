#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _TW>
	struct DocumentPA : public DocumentLDA<_TW>
	{
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;

		tvector<TID> Z2s;
		Eigen::Matrix<WeightType, -1, -1> numByTopic1_2;

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, Z2s);
	};
    
    class IPAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentPA<TermWeight::one>;
		static IPAModel* create(TermWeight _weight, size_t _K1 = 1, size_t _K2 = 1, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t getDirichletEstIteration() const = 0;
		virtual void setDirichletEstIteration(size_t iter) = 0;
		virtual size_t getK2() const = 0;
		virtual FLOAT getSubAlpha(TID k1, TID k2) const = 0;
		virtual std::vector<FLOAT> getSubTopicBySuperTopic(TID k) const = 0;
		virtual std::vector<std::pair<TID, FLOAT>> getSubTopicBySuperTopicSorted(TID k, size_t topN) const = 0;
	};
}