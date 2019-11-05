#include "HLDAModel.hpp"

namespace tomoto
{
	template class HLDAModel<TermWeight::one>;
	template class HLDAModel<TermWeight::idf>;
	template class HLDAModel<TermWeight::pmi>;

	IHLDAModel* IHLDAModel::create(TermWeight _weight, size_t levelDepth, FLOAT _alpha, FLOAT _eta, FLOAT _gamma, const RandGen& _rg)
	{
		SWITCH_TW(_weight, HLDAModel, levelDepth, _alpha, _eta, _gamma, _rg);
	}
}