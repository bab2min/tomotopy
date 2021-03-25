#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentLLDA : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		using WeightType = typename DocumentLDA<_tw>::WeightType;
		Eigen::Matrix<int8_t, -1, 1> labelMask;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, labelMask);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, labelMask);
	};

	class ILLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static ILLDAModel* create(TermWeight _weight, const LDAArgs& args,
			bool scalarRng = false);

		virtual const Dictionary& getTopicLabelDict() const = 0;

		virtual size_t getNumTopicsPerLabel() const = 0;
	};
}