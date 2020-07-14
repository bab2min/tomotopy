#include "DMRModel.hpp"

namespace tomoto
{
	/*template class DMRModel<TermWeight::one>;
	template class DMRModel<TermWeight::idf>;
	template class DMRModel<TermWeight::pmi>;*/

	IDMRModel* IDMRModel::create(TermWeight _weight, size_t _K, Float _defaultAlpha, Float _sigma, Float _eta, Float _alphaEps, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, DMRModel, _K, _defaultAlpha, _sigma, _eta, _alphaEps, seed);
	}
}