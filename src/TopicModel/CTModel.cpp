#include "CTModel.hpp"

namespace tomoto
{
	/*template class CTModel<TermWeight::one>;
	template class CTModel<TermWeight::idf>;
	template class CTModel<TermWeight::pmi>;*/

	ICTModel* ICTModel::create(TermWeight _weight, size_t _K, Float smoothingAlpha, Float _eta, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, CTModel, _K, smoothingAlpha, _eta, seed);
	}
}