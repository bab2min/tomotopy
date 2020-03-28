#pragma once
#include "LDAModel.hpp"
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw, size_t _Flags = 0>
	struct DocumentDTM : public DocumentLDA<_tw, _Flags>
	{
		using BaseDocument = DocumentLDA<_tw, _Flags>;
		using DocumentLDA<_tw, _Flags>::DocumentLDA;
		using WeightType = typename std::conditional<_tw == TermWeight::one, int32_t, float>::type;
	};

    class IDTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDTM<TermWeight::one>;
		static IDTModel* create(TermWeight _weight, size_t _K = 1, Float _alpha = 0.1, Float _eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });
		
	};
}
