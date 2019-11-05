#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _TW, size_t _Flags = 0>
	struct DocumentSLDA : public DocumentLDA<_TW, _Flags>
	{
		using DocumentLDA<_TW, _Flags>::DocumentLDA;
		std::vector<FLOAT> y;
		DEFINE_SERIALIZER_AFTER_BASE2(DocumentLDA<_TW, _Flags>, y);
	};

	class ISLDAModel : public ILDAModel
	{
	public:
		enum class GLM
		{
			linear = 0,
			binary_logistic = 1,
		};

		using DefaultDocType = DocumentSLDA<TermWeight::one>;
		static ISLDAModel* create(TermWeight _weight, size_t _K = 1, 
			const std::vector<ISLDAModel::GLM>& vars = {},
			FLOAT alpha = 0.1, FLOAT _eta = 0.01,
			const std::vector<FLOAT>& _mu = {}, const std::vector<FLOAT>& _nuSq = {},
			const std::vector<FLOAT>& _glmParam = {},
			const RandGen& _rg = RandGen{ std::random_device{}() });

		virtual size_t addDoc(const std::vector<std::string>& words, const std::vector<FLOAT>& y) = 0;
		virtual std::unique_ptr<DocumentBase> makeDoc(const std::vector<std::string>& words, const std::vector<FLOAT>& y) const = 0;
		
		virtual size_t getF() const = 0;
		virtual std::vector<FLOAT> getRegressionCoef(size_t f) const = 0;
		virtual GLM getTypeOfVar(size_t f) const = 0;
		virtual std::vector<FLOAT> estimateVars(const DocumentBase* doc) const = 0;
	};
}