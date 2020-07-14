#include "GDMRModel.hpp"

namespace tomoto
{
	/*template class GDMRModel<TermWeight::one>;
	template class GDMRModel<TermWeight::idf>;
	template class GDMRModel<TermWeight::pmi>;*/

    IGDMRModel* IGDMRModel::create(TermWeight _weight, size_t _K, const std::vector<uint64_t>& degreeByF, 
		Float _defaultAlpha, Float _sigma, Float _sigma0, Float _eta, Float _alphaEps, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, GDMRModel, _K, degreeByF, _defaultAlpha, _sigma, _sigma0, _eta, _alphaEps, seed);
	}
}
