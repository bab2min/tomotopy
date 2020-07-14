#include "LDAModel.hpp"

namespace tomoto
{
	/*template class LDAModel<TermWeight::one>;
	template class LDAModel<TermWeight::idf>;
	template class LDAModel<TermWeight::pmi>;*/

    ILDAModel* ILDAModel::create(TermWeight _weight, size_t _K, Float _alpha, Float _eta, size_t seed, bool scalarRng)
    {
        TMT_SWITCH_TW(_weight, scalarRng, LDAModel, _K, _alpha, _eta, seed);
    }
}
