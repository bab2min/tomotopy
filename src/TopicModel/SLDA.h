#pragma once
#include "LDA.h"

namespace tomoto
{
    template<TermWeight _tw>
	struct DocumentSLDA : public DocumentLDA<_tw>
	{
		using BaseDocument = DocumentLDA<_tw>;
		using DocumentLDA<_tw>::DocumentLDA;
		std::vector<Float> y;
		DEFINE_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 0, y);
		DEFINE_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(BaseDocument, 1, 0x00010001, y);
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
			Float alpha = 0.1, Float _eta = 0.01,
			const std::vector<Float>& _mu = {}, const std::vector<Float>& _nuSq = {},
			const std::vector<Float>& _glmParam = {},
			size_t seed = std::random_device{}(),
			bool scalarRng = false);

		virtual size_t getF() const = 0;
		virtual std::vector<Float> getRegressionCoef(size_t f) const = 0;
		virtual GLM getTypeOfVar(size_t f) const = 0;
		virtual std::vector<Float> estimateVars(const DocumentBase* doc) const = 0;
	};
}