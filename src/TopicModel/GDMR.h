#pragma once
#include "DMR.h"

namespace tomoto
{
    template<TermWeight _TW, bool _Shared = false>
	struct DocumentGDMR : public DocumentDMR<_TW, _Shared>
	{
		using DocumentDMR<_TW, _Shared>::DocumentDMR;
		std::vector<FLOAT> metadataC;

		DEFINE_SERIALIZER_AFTER_BASE2(DocumentDMR<_TW, _Shared>, metadataC);
	};

    class IGDMRModel : public IDMRModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IGDMRModel* create(TermWeight _weight, size_t _K = 1, const std::vector<size_t>& _degreeByF = {},
			FLOAT defaultAlpha = 1.0, FLOAT _sigma = 1.0, FLOAT _eta = 0.01, FLOAT _alphaEps = 1e-10,
			const RANDGEN& _rg = RANDGEN{ std::random_device{}() });

		virtual FLOAT getSigma0() const = 0;
		virtual void setSigma0(FLOAT) = 0;
		virtual const std::vector<size_t>& getFs() const = 0;
		virtual std::vector<FLOAT> getLambdaByTopic(TID tid) const = 0;
		virtual void setMdRange(const std::vector<FLOAT>& vMin, const std::vector<FLOAT>& vMax) = 0;
	};
}