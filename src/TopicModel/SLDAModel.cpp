#include "SLDAModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentSLDA, BaseDocument, 0, y);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentSLDA, BaseDocument, 1, 0x00010001, y);

	TMT_INSTANTIATE_DOC(DocumentSLDA);

    ISLDAModel* ISLDAModel::create(TermWeight _weight, const SLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, SLDAModel, args);
	}
}