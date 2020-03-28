#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentHLDA : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using WeightType = typename DocumentLDA<_tw>::WeightType;
		using DocumentLDA<_tw>::DocumentLDA;

		// numByTopic indicates numByLevel in HLDAModel.
		// Zs indicates level in HLDAModel.
		std::vector<int32_t> path;

		template<typename _TopicModel> void update(WeightType* ptr, const _TopicModel& mdl);

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, path);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, path);
	};

	class IHLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentHLDA<TermWeight::one>;
		static IHLDAModel* create(TermWeight _weight, size_t levelDepth = 1, Float alpha = 0.1, Float eta = 0.01, Float gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual Float getGamma() const = 0;
		virtual size_t getLiveK() const = 0;
		virtual size_t getLevelDepth() const = 0;
		virtual bool isLiveTopic(Tid tid) const = 0;
		virtual size_t getNumDocsOfTopic(Tid tid) const = 0;
		virtual size_t getLevelOfTopic(Tid tid) const = 0;
		virtual size_t getParentTopicId(Tid tid) const = 0;
		virtual std::vector<size_t> getChildTopicId(Tid tid) const = 0;
	};
}
