#include "GDMRModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentGDMR, BaseDocument, 0, metadataOrg);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentGDMR, BaseDocument, 1, 0x00010001, metadataOrg, metadataNormalized);

	TMT_INSTANTIATE_DOC(DocumentGDMR);

    IGDMRModel* IGDMRModel::create(TermWeight _weight, const GDMRArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, GDMRModel, args);
	}
}
