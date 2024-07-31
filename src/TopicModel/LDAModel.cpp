#include "LDAModel.hpp"

namespace tomoto
{
    DEFINE_OUT_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentLDA, DocumentBase, 0, Zs, wordWeights);
    DEFINE_OUT_TAGGED_SERIALIZER_AFTER_BASE_WITH_VERSION(DocumentLDA, DocumentBase, 1, 0x00010001, Zs, wordWeights);

    TMT_INSTANTIATE_DOC(DocumentLDA);

    ILDAModel* ILDAModel::create(TermWeight _weight, const LDAArgs& args, bool scalarRng)
    {
        TMT_SWITCH_TW(_weight, scalarRng, LDAModel, args);
    }
}
