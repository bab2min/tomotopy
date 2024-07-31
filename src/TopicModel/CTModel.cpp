#include "CTModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentCTM, BaseDocument, 0, smBeta);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentCTM, BaseDocument, 1, 0x00010001, smBeta);

	TMT_INSTANTIATE_DOC(DocumentCTM);

	ICTModel* ICTModel::create(TermWeight _weight, const CTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, CTModel, args);
	}
}