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

		RawDoc::MiscType makeMisc(const ITopicModel* tm) const override
		{
			RawDoc::MiscType ret = DocumentDMR<_tw>::makeMisc(tm);
			ret["numeric_metadata"] = metadataOrg;
			return ret;
		}

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, metadataOrg);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, metadataOrg, metadataNormalized);
	};

	struct GDMRArgs : public DMRArgs
	{
		std::vector<uint64_t> degrees;
		Float sigma0 = 3.0;
		Float orderDecay = 0;
	};

    class IGDMRModel : public IDMRModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IGDMRModel* create(TermWeight _weight, const GDMRArgs& args,
			bool scalarRng = false);

		virtual Float getSigma0() const = 0;
		virtual Float getOrderDecay() const = 0;
		virtual void setSigma0(Float) = 0;
		virtual const std::vector<uint64_t>& getFs() const = 0;
		virtual std::vector<Float> getLambdaByTopic(Tid tid) const = 0;

		virtual std::vector<Float> getTDF(
			const Float* metadata, 
			const std::string& metadataCat, 
			const std::vector<std::string>& multiMetadataCat, 
			bool normalize
		) const = 0;

		virtual std::vector<Float> getTDFBatch(
			const Float* metadata, 
			const std::string& metadataCat, 
			const std::vector<std::string>& multiMetadataCat,
			size_t stride, 
			size_t cnt, 
			bool normalize
		) const = 0;

		virtual void setMdRange(const std::vector<Float>& vMin, const std::vector<Float>& vMax) = 0;
		virtual void getMdRange(std::vector<Float>& vMin, std::vector<Float>& vMax) const = 0;
	};
}
