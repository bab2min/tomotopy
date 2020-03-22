#include "PAModel.hpp"

namespace tomoto
{
	template class PAModel<TermWeight::one>;
	template class PAModel<TermWeight::idf>;
	template class PAModel<TermWeight::pmi>;

	IPAModel* IPAModel::create(TermWeight _weight, size_t _K, size_t _K2, Float _alpha, Float _eta, const RandGen& _rg)
	{
		SWITCH_TW(_weight, PAModel, _K, _K2, _alpha, _eta, _rg);
	}
}
