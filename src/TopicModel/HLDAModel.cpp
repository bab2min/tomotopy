#include "HLDAModel.hpp"

namespace tomoto
{
	/*template class HLDAModel<TermWeight::one>;
	template class HLDAModel<TermWeight::idf>;
	template class HLDAModel<TermWeight::pmi>;*/

	IHLDAModel* IHLDAModel::create(TermWeight _weight, size_t levelDepth, Float _alpha, Float _eta, Float _gamma, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HLDAModel, levelDepth, _alpha, _eta, _gamma, seed);
	}
}