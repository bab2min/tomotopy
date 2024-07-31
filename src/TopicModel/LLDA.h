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

		DECLARE_SERIALIZER_WITH_VERSION(0);
		DECLARE_SERIALIZER_WITH_VERSION(1);
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