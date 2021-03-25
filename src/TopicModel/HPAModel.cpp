#include "HPAModel.hpp"

namespace tomoto
{
    IHPAModel* IHPAModel::create(TermWeight _weight, bool _exclusive, const HPAArgs& args, bool scalarRng)
	{
		if (_exclusive)
		{
			//TMT_SWITCH_TW(_weight, HPAModelExclusive, _K, _K2, _alphaSum, _eta, seed);
		}
		else
		{
			TMT_SWITCH_TW(_weight, scalarRng, HPAModel, args);
		}
		return nullptr;
	}
}
