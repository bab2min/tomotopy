#include "CTModel.hpp"

namespace tomoto
{
	template class CTModel<TermWeight::one>;
	template class CTModel<TermWeight::idf>;
	template class CTModel<TermWeight::pmi>;

	ICTModel* ICTModel::create(TermWeight _weight, size_t _K, FLOAT smoothingAlpha, FLOAT _eta, const RandGen& _rg)
	{
		SWITCH_TW(_weight, CTModel, _K, smoothingAlpha, _eta, _rg);
	}
}