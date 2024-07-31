#include "PAModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentPA, BaseDocument, 0, Z2s);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentPA, BaseDocument, 1, 0x00010001, Z2s);

	TMT_INSTANTIATE_DOC(DocumentPA);

	IPAModel* IPAModel::create(TermWeight _weight, const PAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PAModel, args);
	}
}
