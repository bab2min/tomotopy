#include "MGLDAModel.hpp"

namespace tomoto
{
	DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentMGLDA, BaseDocument, 0, sents, Vs, numGl, numBySentWin, numByWinL, numByWin, numByWinTopicL);
	DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentMGLDA, BaseDocument, 1, 0x00010001, sents, Vs, numGl, numBySentWin, numByWinL, numByWin, numByWinTopicL);

	TMT_INSTANTIATE_DOC(DocumentMGLDA);

    IMGLDAModel* IMGLDAModel::create(TermWeight _weight, const MGLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, MGLDAModel, args);
	}
}