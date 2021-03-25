#include "HLDAModel.hpp"

namespace tomoto
{
	IHLDAModel* IHLDAModel::create(TermWeight _weight, const HLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HLDAModel, args);
	}
}