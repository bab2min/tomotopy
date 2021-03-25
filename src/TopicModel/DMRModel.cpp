#include "DMRModel.hpp"

namespace tomoto
{
	IDMRModel* IDMRModel::create(TermWeight _weight, const DMRArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, DMRModel, args);
	}
}