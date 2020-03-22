#include "HPAModel.hpp"

namespace tomoto
{
	template class HPAModel<TermWeight::one>;
	template class HPAModel<TermWeight::idf>;
	template class HPAModel<TermWeight::pmi>;

    IHPAModel* IHPAModel::create(TermWeight _weight, bool _exclusive, size_t _K, size_t _K2, Float _alphaSum, Float _eta, const RandGen& _rg)
	{
		if (_exclusive)
		{
			//SWITCH_TW(_weight, HPAModelExclusive, _K, _K2, _alphaSum, _eta, _rg);
		}
		else
		{
			SWITCH_TW(_weight, HPAModel, _K, _K2, _alphaSum, _eta, _rg);
		}
		return nullptr;
	}
}
