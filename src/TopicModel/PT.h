#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentPTM : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		using WeightType = typename DocumentLDA<_tw>::WeightType;
		
		uint64_t pseudoDoc = 0;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, pseudoDoc);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, pseudoDoc);
	};

	class IPTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentPTM<TermWeight::one>;
		static IPTModel* create(TermWeight _weight, size_t _K = 1, size_t _P = 100, 
			Float alpha = 0.1, Float eta = 0.01, Float lambda = 0.01, size_t seed = std::random_device{}(),
			bool scalarRng = false);
	};
}