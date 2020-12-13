#pragma once
#include "LDAModel.hpp"
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentDTM : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;

		uint64_t timepoint = 0;
		ShareableVector<Float> eta;
		sample::AliasMethod<> aliasTable;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, timepoint);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, timepoint);
	};

    class IDTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDTM<TermWeight::one>;
		static IDTModel* create(TermWeight _weight, size_t _K = 1, size_t _T = 1, 
			Float _alphaVar = 1.0, Float _etaVar = 1.0, Float _phiVar = 1.0, 
			Float _shapeA = 0.03, Float _shapeB = 0.1, Float _shapeC = 0.55,
			Float _etaRegL2 = 0,
			size_t seed = std::random_device{}(),
			bool scalarRng = false);
		
		virtual size_t getT() const = 0;
		virtual std::vector<uint32_t> getNumDocsByT() const = 0;

		virtual Float getAlphaVar() const = 0;
		virtual Float getEtaVar() const = 0;
		virtual Float getPhiVar() const = 0;

		virtual Float getShapeA() const = 0;
		virtual Float getShapeB() const = 0;
		virtual Float getShapeC() const = 0;

		virtual void setShapeA(Float a) = 0;
		virtual void setShapeB(Float a) = 0;
		virtual void setShapeC(Float a) = 0;

		virtual Float getAlpha(size_t k, size_t t) const = 0;
		virtual std::vector<Float> getPhi(size_t k, size_t t) const = 0;
	};
}
