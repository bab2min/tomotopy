#include "PLDAModel.hpp"

namespace tomoto
{
	std::unique_ptr<IPLDAModel> IPLDAModel::create(TermWeight _weight, const PLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PLDAModel, args);
	}
}
