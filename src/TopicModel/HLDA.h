#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _TW>
	struct DocumentHLDA : public DocumentLDA<_TW>
	{
		using WeightType = typename DocumentLDA<_TW>::WeightType;
		using DocumentLDA<_TW>::DocumentLDA;

		// numByTopic indicates numByLevel in HLDAModel.
		// Zs indicates level in HLDAModel.
		std::vector<int32_t> path;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, path);
	};

	class IHLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentHLDA<TermWeight::one>;
		static IHLDAModel* create(TermWeight _weight, size_t levelDepth = 1, FLOAT alpha = 0.1, FLOAT eta = 0.01, FLOAT gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual FLOAT getGamma() const = 0;
		virtual size_t getLiveK() const = 0;
		virtual size_t getLevelDepth() const = 0;
		virtual bool isLiveTopic(TID tid) const = 0;
		virtual size_t getNumDocsOfTopic(TID tid) const = 0;
		virtual size_t getLevelOfTopic(TID tid) const = 0;
		virtual size_t getParentTopicId(TID tid) const = 0;
		virtual std::vector<size_t> getChildTopicId(TID tid) const = 0;
	};
}
