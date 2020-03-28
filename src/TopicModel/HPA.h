#pragma once
#include "PA.h"

namespace tomoto
{
    template<TermWeight _tw>
	struct DocumentHPA : public DocumentPA<_tw>
	{
		using BaseDocument = DocumentPA<_tw>;
		using DocumentPA<_tw>::DocumentPA;
		using WeightType = typename DocumentPA<_tw>::WeightType;

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);

		DEFINE_SERIALIZER_BASE_WITH_VERSION(BaseDocument, 0);
		DEFINE_SERIALIZER_BASE_WITH_VERSION(BaseDocument, 1);
	};

	class IHPAModel : public IPAModel
	{
	public:
		using DefaultDocType = DocumentHPA<TermWeight::one>;
		static IHPAModel* create(TermWeight _weight, bool _exclusive = false, size_t _K1 = 1, size_t _K2 = 1, Float _alpha = 50, Float _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });
	};
}
