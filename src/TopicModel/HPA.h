#pragma once
#include "PA.h"

namespace tomoto
{
    template<TermWeight _TW>
	struct DocumentHPA : public DocumentPA<_TW>
	{
		using DocumentPA<_TW>::DocumentPA;
		using WeightType = typename DocumentPA<_TW>::WeightType;

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);
	};

	class IHPAModel : public IPAModel
	{
	public:
		using DefaultDocType = DocumentHPA<TermWeight::one>;
		static IHPAModel* create(TermWeight _weight, bool _exclusive = false, size_t _K1 = 1, size_t _K2 = 1, FLOAT _alpha = 50, FLOAT _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });
	};
}