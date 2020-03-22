#include "LDAModel.hpp"

namespace tomoto
{
	template class LDAModel<TermWeight::one>;
	template class LDAModel<TermWeight::idf>;
	template class LDAModel<TermWeight::pmi>;

    ILDAModel* ILDAModel::create(TermWeight _weight, size_t _K, Float _alpha, Float _eta, const RandGen& _rg)
    {
        SWITCH_TW(_weight, LDAModel, _K, _alpha, _eta, _rg);
    }
}
