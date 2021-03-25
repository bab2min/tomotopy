#include "DTModel.hpp"

namespace tomoto
{
	IDTModel* IDTModel::create(TermWeight _weight, const DTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, DTModel, args);
	}
}
