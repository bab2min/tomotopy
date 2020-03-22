#include "GDMRModel.hpp"

namespace tomoto
{
	template class GDMRModel<TermWeight::one>;
	template class GDMRModel<TermWeight::idf>;
	template class GDMRModel<TermWeight::pmi>;

    IGDMRModel* IGDMRModel::create(TermWeight _weight, size_t _K, const std::vector<size_t>& degreeByF, Float _defaultAlpha, Float _sigma, Float _eta, Float _alphaEps, const RandGen& _rg)
	{
		SWITCH_TW(_weight, GDMRModel, _K, degreeByF, _defaultAlpha, _sigma, _eta, _alphaEps, _rg);
	}
}
