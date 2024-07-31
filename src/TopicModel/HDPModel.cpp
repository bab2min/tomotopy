#include "HDPModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentHDP, BaseDocument, 0, numTopicByTable);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentHDP, BaseDocument, 1, 0x00010001, numTopicByTable);

	TMT_INSTANTIATE_DOC(DocumentHDP);

    IHDPModel* IHDPModel::create(TermWeight _weight, const HDPArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HDPModel, args);
	}
}