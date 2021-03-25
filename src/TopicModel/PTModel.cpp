#include "PTModel.hpp"

namespace tomoto
{
	IPTModel* IPTModel::create(TermWeight _weight, const PTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PTModel, args);
	}
}