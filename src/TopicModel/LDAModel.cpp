#include "LDAModel.hpp"

namespace tomoto
{
    ILDAModel* ILDAModel::create(TermWeight _weight, const LDAArgs& args, bool scalarRng)
    {
        TMT_SWITCH_TW(_weight, scalarRng, LDAModel, args);
    }
}
