#include "HLDAModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentHLDA, BaseDocument, 0, path);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentHLDA, BaseDocument, 1, 0x00010001, path);

	TMT_INSTANTIATE_DOC(DocumentHLDA);

	IHLDAModel* IHLDAModel::create(TermWeight _weight, const HLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HLDAModel, args);
	}
}