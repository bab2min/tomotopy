#include "LLDAModel.hpp"

namespace tomoto
{
	/*template class LLDAModel<TermWeight::one>;
	template class LLDAModel<TermWeight::idf>;
	template class LLDAModel<TermWeight::pmi>;*/

	ILLDAModel* ILLDAModel::create(TermWeight _weight, size_t _K, Float _alpha, Float _eta, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, LLDAModel, _K, _alpha, _eta, seed);
	}
}