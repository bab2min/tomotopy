#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _tw>
	struct DocumentLLDA : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		using WeightType = typename DocumentLDA<_tw>::WeightType;
		Eigen::Matrix<int8_t, -1, 1> labelMask;

		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, labelMask);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, labelMask);
	};

	class ILLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static ILLDAModel* create(TermWeight _weight, size_t _K = 1, Float alpha = 0.1, Float eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& label) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& label) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer, 
			const std::vector<std::string>& label) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const RawDocTokenizer::Factory& tokenizer,
			const std::vector<std::string>& label) const = 0;

		virtual size_t addDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len, 
			const std::vector<std::string>& label) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::string& rawStr, const std::vector<Vid>& words,
			const std::vector<uint32_t>& pos, const std::vector<uint16_t>& len,
			const std::vector<std::string>& label) const = 0;

		virtual const Dictionary& getTopicLabelDict() const = 0;

		virtual size_t getNumTopicsPerLabel() const = 0;
	};
}