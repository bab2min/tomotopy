#include "SLDAModel.hpp"

namespace tomoto
{
	template class SLDAModel<TermWeight::one>;
	template class SLDAModel<TermWeight::idf>;
	template class SLDAModel<TermWeight::pmi>;

    ISLDAModel* ISLDAModel::create(TermWeight _weight, size_t _K, const std::vector<ISLDAModel::GLM>& vars,
		FLOAT _alpha, FLOAT _eta, 
		const std::vector<FLOAT>& _mu, const std::vector<FLOAT>& _nuSq,
		const std::vector<FLOAT>& _glmParam,
		const RandGen& _rg)
	{
		SWITCH_TW(_weight, SLDAModel, _K, vars, _alpha, _eta, _mu, _nuSq, _glmParam, _rg);
	}
}