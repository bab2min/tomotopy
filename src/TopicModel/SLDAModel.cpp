#include "SLDAModel.hpp"

namespace tomoto
{
    ISLDAModel* ISLDAModel::create(TermWeight _weight, const SLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, SLDAModel, args);
	}
}