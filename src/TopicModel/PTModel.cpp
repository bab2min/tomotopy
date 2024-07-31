#include "PTModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentPT, BaseDocument, 0, pseudoDoc);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentPT, BaseDocument, 1, 0x00010001, pseudoDoc);

	TMT_INSTANTIATE_DOC(DocumentPT);

	IPTModel* IPTModel::create(TermWeight _weight, const PTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PTModel, args);
	}
}