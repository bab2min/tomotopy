#include "MGLDAModel.hpp"

namespace tomoto
{
	template class MGLDAModel<TermWeight::one>;
	template class MGLDAModel<TermWeight::idf>;
	template class MGLDAModel<TermWeight::pmi>;

    IMGLDAModel* IMGLDAModel::create(TermWeight _weight, size_t _KG, size_t _KL, size_t _T,
		FLOAT _alphaG, FLOAT _alphaL, FLOAT _alphaMG, FLOAT _alphaML,
		FLOAT _etaG, FLOAT _etaL, FLOAT _gamma, const RandGen& _rg)
	{
		SWITCH_TW(_weight, MGLDAModel, _KG, _KL, _T,
			_alphaG, _alphaL, _alphaMG, _alphaML,
			_etaG, _etaL, _gamma, _rg);
	}
}