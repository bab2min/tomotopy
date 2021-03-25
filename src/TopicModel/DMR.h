#pragma once
#include "LDA.h"

namespace tomoto
{
	class IDMRModel;

	template<TermWeight _tw>
	struct DocumentDMR : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		uint64_t metadata = 0;

		RawDoc::MiscType makeMisc(const ITopicModel* tm) const override;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, metadata);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, metadata);
	};

	struct DMRArgs : public LDAArgs
	{
		Float alphaEps = 1e-10;
		Float sigma = 1.0;
	};

    class IDMRModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IDMRModel* create(TermWeight _weight, const DMRArgs& args,
			bool scalarRng = false);

		virtual void setAlphaEps(Float _alphaEps) = 0;
		virtual Float getAlphaEps() const = 0;
		virtual void setOptimRepeat(size_t repeat) = 0;
		virtual size_t getOptimRepeat() const = 0;
		virtual size_t getF() const = 0;
		virtual Float getSigma() const = 0;
		virtual const Dictionary& getMetadataDict() const = 0;
		virtual std::vector<Float> getLambdaByMetadata(size_t metadataId) const = 0;
		virtual std::vector<Float> getLambdaByTopic(Tid tid) const = 0;
	};

	template<TermWeight _tw>
	RawDoc::MiscType DocumentDMR<_tw>::makeMisc(const ITopicModel* tm) const
	{
		RawDoc::MiscType ret = DocumentLDA<_tw>::makeMisc(tm);
		auto inst = static_cast<const IDMRModel*>(tm);
		ret["metadata"] = inst->getMetadataDict().toWord(metadata);
		return ret;
	}
}