#include "PLDAModel.hpp"

namespace tomoto
{
	/*template class PLDAModel<TermWeight::one>;
	template class PLDAModel<TermWeight::idf>;
	template class PLDAModel<TermWeight::pmi>;*/

	IPLDAModel* IPLDAModel::create(TermWeight _weight, size_t _numLatentTopics, size_t _numTopicsPerLabel, Float _alpha, Float _eta, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PLDAModel, _numLatentTopics, _numTopicsPerLabel, _alpha, _eta, seed);
	}
}
