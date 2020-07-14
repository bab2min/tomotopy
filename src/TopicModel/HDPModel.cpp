#include "HDPModel.hpp"

namespace tomoto
{
	/*template class HDPModel<TermWeight::one>;
	template class HDPModel<TermWeight::idf>;
	template class HDPModel<TermWeight::pmi>;*/

    IHDPModel* IHDPModel::create(TermWeight _weight, size_t _K, Float _alpha , Float _eta, Float _gamma, size_t seed, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HDPModel, _K, _alpha, _eta, _gamma, seed);
	}
}