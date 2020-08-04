#pragma once
#include "DMR.h"

namespace tomoto
{
    template<TermWeight _tw>
	struct DocumentGDMR : public DocumentDMR<_tw>
	{
		using BaseDocument = DocumentDMR<_tw>;
		using DocumentDMR<_tw>::DocumentDMR;
		std::vector<Float> metadataOrg, metadataNormalized;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, metadataOrg);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, metadataOrg, metadataNormalized);
	};

    class IGDMRModel : public IDMRModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IGDMRModel* create(TermWeight _weight, size_t _K = 1, const std::vector<uint64_t>& _degreeByF = {},
			Float defaultAlpha = 1.0, Float _sigma = 1.0, Float _sigma0 = 1.0, Float _eta = 0.01, Float _alphaEps = 1e-10,
			size_t seed = std::random_device{}(),
			bool scalarRng = false);

		virtual Float getSigma0() const = 0;
		virtual void setSigma0(Float) = 0;
		virtual const std::vector<uint64_t>& getFs() const = 0;
		virtual std::vector<Float> getLambdaByTopic(Tid tid) const = 0;

		virtual std::vector<Float> getTDF(const Float* metadata, bool normalize) const = 0;
		virtual std::vector<Float> getTDFBatch(const Float* metadata, size_t stride, size_t cnt, bool normalize) const = 0;

		virtual void setMdRange(const std::vector<Float>& vMin, const std::vector<Float>& vMax) = 0;
		virtual void getMdRange(std::vector<Float>& vMin, std::vector<Float>& vMax) const = 0;
	};
}
