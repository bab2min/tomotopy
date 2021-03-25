#include "PAModel.hpp"

namespace tomoto
{
	IPAModel* IPAModel::create(TermWeight _weight, const PAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PAModel, args);
	}
}
