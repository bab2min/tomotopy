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

	struct HPAArgs : public PAArgs
	{
	};

	class IHPAModel : public IPAModel
	{
	public:
		using DefaultDocType = DocumentHPA<TermWeight::one>;
		static IHPAModel* create(TermWeight _weight, bool _exclusive, const HPAArgs& args,
			bool scalarRng = false);
	};
}
