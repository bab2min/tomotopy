#include "DTModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentDTM, BaseDocument, 0, timepoint);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentDTM, BaseDocument, 1, 0x00010001, timepoint);

	TMT_INSTANTIATE_DOC(DocumentDTM);

	IDTModel* IDTModel::create(TermWeight _weight, const DTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, DTModel, args);
	}
}
