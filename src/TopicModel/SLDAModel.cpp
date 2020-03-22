#include "SLDAModel.hpp"

namespace tomoto
{
	template class SLDAModel<TermWeight::one>;
	template class SLDAModel<TermWeight::idf>;
	template class SLDAModel<TermWeight::pmi>;

    ISLDAModel* ISLDAModel::create(TermWeight _weight, size_t _K, const std::vector<ISLDAModel::GLM>& vars,
		Float _alpha, Float _eta, 
		const std::vector<Float>& _mu, const std::vector<Float>& _nuSq,
		const std::vector<Float>& _glmParam,
		const RandGen& _rg)
	{
		SWITCH_TW(_weight, SLDAModel, _K, vars, _alpha, _eta, _mu, _nuSq, _glmParam, _rg);
	}
}