#include "HPAModel.hpp"

namespace tomoto
{
	/*template class HPAModel<TermWeight::one>;
	template class HPAModel<TermWeight::idf>;
	template class HPAModel<TermWeight::pmi>;*/

    IHPAModel* IHPAModel::create(TermWeight _weight, bool _exclusive, size_t _K, size_t _K2, Float _alphaSum, Float _eta, size_t seed, bool scalarRng)
	{
		if (_exclusive)
		{
			//TMT_SWITCH_TW(_weight, HPAModelExclusive, _K, _K2, _alphaSum, _eta, seed);
		}
		else
		{
			TMT_SWITCH_TW(_weight, scalarRng, HPAModel, _K, _K2, _alphaSum, _eta, seed);
		}
		return nullptr;
	}
}
