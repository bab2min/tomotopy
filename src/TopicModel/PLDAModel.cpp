#include "PLDAModel.hpp"

namespace tomoto
{
	IPLDAModel* IPLDAModel::create(TermWeight _weight, const PLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PLDAModel, args);
	}
}
