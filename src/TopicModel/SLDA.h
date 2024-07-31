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

		RawDoc::MiscType makeMisc(const ITopicModel* tm) const override
		{
			RawDoc::MiscType ret = DocumentLDA<_tw>::makeMisc(tm);
			ret["y"] = y;
			return ret;
		}
		
		DECLARE_SERIALIZER_WITH_VERSION(0);
		DECLARE_SERIALIZER_WITH_VERSION(1);
	};

	struct SLDAArgs;

	class ISLDAModel : public ILDAModel
	{
	public:
		enum class GLM
		{
			linear = 0,
			binary_logistic = 1,
		};

		using DefaultDocType = DocumentSLDA<TermWeight::one>;
		static ISLDAModel* create(TermWeight _weight, const SLDAArgs& args,
			bool scalarRng = false);

		virtual size_t getF() const = 0;
		virtual std::vector<Float> getRegressionCoef(size_t f) const = 0;
		virtual GLM getTypeOfVar(size_t f) const = 0;
		virtual std::vector<Float> estimateVars(const DocumentBase* doc) const = 0;
	};

	struct SLDAArgs : public LDAArgs
	{
		std::vector<ISLDAModel::GLM> vars;
		std::vector<Float> mu;
		std::vector<Float> nuSq;
		std::vector<Float> glmParam;
	};
}