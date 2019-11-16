#pragma once
#include "LDA.h"

namespace tomoto
{
	template<TermWeight _TW>
	struct DocumentLLDA : public DocumentLDA<_TW>
	{
		using DocumentLDA<_TW>::DocumentLDA;
		using WeightType = typename DocumentLDA<_TW>::WeightType;
		Eigen::Matrix<int8_t, -1, 1> labelMask;

		DEFINE_SERIALIZER_AFTER_BASE(DocumentLDA<_TW>, labelMask);
	};

	class ILLDAModel : public ILDAModel
	{
	public:
		using DefaultDocType = DocumentLLDA<TermWeight::one>;
		static ILLDAModel* create(TermWeight _weight, size_t _K = 1, FLOAT alpha = 0.1, FLOAT eta = 0.01, const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<std::string>& label) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<std::string>& label) const = 0;

		virtual const Dictionary& getTopicLabelDict() const = 0;

		virtual size_t getNumTopicsPerLabel() const = 0;
	};
}