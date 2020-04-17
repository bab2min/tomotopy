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

		size_t timepoint = 0;
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
			const RandGen& _rg = RandGen{ std::random_device{}() });
		
		virtual size_t addDoc(const std::vector<std::string>& words, size_t timepoint) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, size_t timepoint) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			size_t timepoint) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			size_t timepoint) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			size_t timepoint) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			size_t timepoint) const = 0;

		virtual size_t getT() const = 0;
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
