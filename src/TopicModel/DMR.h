#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _TW, size_t _Flags = 0>
	struct DocumentDMR : public DocumentLDA<_TW, _Flags>
	{
		using DocumentLDA<_TW, _Flags>::DocumentLDA;
		size_t metadata = 0;

		DEFINE_SERIALIZER_AFTER_BASE2(DocumentLDA<_TW, _Flags>, metadata);
	};

    class IDMRModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentDMR<TermWeight::one>;
		static IDMRModel* create(TermWeight _weight, size_t _K = 1,
			FLOAT defaultAlpha = 1.0, FLOAT _sigma = 1.0, FLOAT _eta = 0.01, FLOAT _alphaEps = 1e-10,
			const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& metadata) const = 0;
		
		virtual void setAlphaEps(FLOAT _alphaEps) = 0;
		virtual FLOAT getAlphaEps() const = 0;
		virtual void setOptimRepeat(size_t repeat) = 0;
		virtual size_t getOptimRepeat() const = 0;
		virtual size_t getF() const = 0;
		virtual FLOAT getSigma() const = 0;
		virtual const Dictionary& getMetadataDict() const = 0;
		virtual std::vector<FLOAT> getLambdaByMetadata(size_t metadataId) const = 0;
		virtual std::vector<FLOAT> getLambdaByTopic(TID tid) const = 0;
	};
}