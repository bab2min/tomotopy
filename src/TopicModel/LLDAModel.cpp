#include "LLDAModel.hpp"

namespace tomoto
{
	template class LLDAModel<TermWeight::one>;
	template class LLDAModel<TermWeight::idf>;
	template class LLDAModel<TermWeight::pmi>;

	ILLDAModel* ILLDAModel::create(TermWeight _weight, size_t _K, Float _alpha, Float _eta, const RandGen& _rg)
	{
		SWITCH_TW(_weight, LLDAModel, _K, _alpha, _eta, _rg);
	}
}