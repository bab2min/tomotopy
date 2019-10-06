#include "LLDAModel.hpp"

namespace tomoto
{
	template class LLDAModel<TermWeight::one>;
	template class LLDAModel<TermWeight::idf>;
	template class LLDAModel<TermWeight::pmi>;

	ILLDAModel* ILLDAModel::create(TermWeight _weight, size_t _K, FLOAT _alpha, FLOAT _eta, const RANDGEN& _rg)
	{
		SWITCH_TW(_weight, LLDAModel, _K, _alpha, _eta, _rg);
	}
}