#include "DMRModel.hpp"

namespace tomoto
{
	template class DMRModel<TermWeight::one>;
	template class DMRModel<TermWeight::idf>;
	template class DMRModel<TermWeight::pmi>;

	IDMRModel* IDMRModel::create(TermWeight _weight, size_t _K, Float _defaultAlpha, Float _sigma, Float _eta, Float _alphaEps, const RandGen& _rg)
	{
		SWITCH_TW(_weight, DMRModel, _K, _defaultAlpha, _sigma, _eta, _alphaEps, _rg);
	}
}