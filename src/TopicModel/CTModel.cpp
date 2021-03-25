#include "CTModel.hpp"

namespace tomoto
{
	ICTModel* ICTModel::create(TermWeight _weight, const CTArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, CTModel, args);
	}
}