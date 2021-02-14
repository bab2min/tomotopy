#include "PTModel.hpp"

namespace tomoto
{

	IPTModel* IPTModel::create(TermWeight _weight, size_t _K, size_t _P, Float _alpha, Float _eta, Float _lambda, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, PTModel, _K, _P, _alpha, _eta, _lambda, seed);
	}
}