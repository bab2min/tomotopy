#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw, size_t _Flags = 0>
	struct DocumentDMR : public DocumentLDA<_tw, _Flags>
	{
		using BaseDocument = DocumentLDA<_tw, _Flags>;
		using DocumentLDA<_tw, _Flags>::DocumentLDA;
		size_t metadata = 0;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, metadata);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, metadata);
	};

    class IDMRModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IDMRModel* create(TermWeight _weight, size_t _K = 1,
			Float defaultAlpha = 1.0, Float _sigma = 1.0, Float _eta = 0.01, Float _alphaEps = 1e-10,
			const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const = 0;
		
		virtual size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<std::string>& metadata) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<std::string>& metadata) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<std::string>& metadata) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<std::string>& metadata) const = 0;

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