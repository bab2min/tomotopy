#include "PLDAModel.hpp"

namespace tomoto
{
	template class PLDAModel<TermWeight::one>;
	template class PLDAModel<TermWeight::idf>;
	template class PLDAModel<TermWeight::pmi>;

	IPLDAModel* IPLDAModel::create(TermWeight _weight, size_t _numLatentTopics, size_t _numTopicsPerLabel, FLOAT _alpha, FLOAT _eta, const RandGen& _rg)
	{
		SWITCH_TW(_weight, PLDAModel, _numLatentTopics, _numTopicsPerLabel, _alpha, _eta, _rg);
	}
}
