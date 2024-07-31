#include "LLDAModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentLLDA, BaseDocument, 0, labelMask);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentLLDA, BaseDocument, 1, 0x00010001, labelMask);

	TMT_INSTANTIATE_DOC(DocumentLLDA);

	ILLDAModel* ILLDAModel::create(TermWeight _weight, const LDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, LLDAModel, args);
	}
}