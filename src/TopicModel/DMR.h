#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentDMR : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		uint64_t metadata = 0;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, metadata);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, metadata);
	};

    class IDMRModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IDMRModel* create(TermWeight _weight, size_t _K = 1,
			Float defaultAlpha = 1.0, Float _sigma = 1.0, Float _eta = 0.01, Float _alphaEps = 1e-10,
			size_t seed = std::random_device{}(),
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
}