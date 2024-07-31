#include "DMRModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentDMR, BaseDocument, 0, metadata);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentDMR, BaseDocument, 1, 0x00010001, metadata, multiMetadata);

	TMT_INSTANTIATE_DOC(DocumentDMR);

	IDMRModel* IDMRModel::create(TermWeight _weight, const DMRArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, DMRModel, args);
	}
}