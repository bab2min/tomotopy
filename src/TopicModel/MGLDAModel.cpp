#include "MGLDAModel.hpp"

namespace tomoto
{
	/*template class MGLDAModel<TermWeight::one>;
	template class MGLDAModel<TermWeight::idf>;
	template class MGLDAModel<TermWeight::pmi>;*/

    IMGLDAModel* IMGLDAModel::create(TermWeight _weight, size_t _KG, size_t _KL, size_t _T,
		Float _alphaG, Float _alphaL, Float _alphaMG, Float _alphaML,
		Float _etaG, Float _etaL, Float _gamma, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, MGLDAModel, _KG, _KL, _T,
			_alphaG, _alphaL, _alphaMG, _alphaML,
			_etaG, _etaL, _gamma, seed);
	}
}