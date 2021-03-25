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
		ShareableMatrix<Float, -1, 1> eta;
		sample::AliasMethod<> aliasTable;

		RawDoc::MiscType makeMisc(const ITopicModel* tm) const override
		{
			RawDoc::MiscType ret = DocumentLDA<_tw>::makeMisc(tm);
			ret["timepoint"] = (uint32_t)timepoint;
			return ret;
		}

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, timepoint);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, timepoint);
	};

	struct DTArgs : public LDAArgs
	{
		size_t t = 1;
		Float phi = 0.1;
		Float shapeA = 0.01;
		Float shapeB = 0.1;
		Float shapeC = 0.55;
		Float etaL2Reg = 0;

		DTArgs()
		{
			alpha[0] = 0.1;
			eta = 0.1;
		}
	};

    class IDTModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDTM<TermWeight::one>;
		static IDTModel* create(TermWeight _weight, const DTArgs& args,
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
