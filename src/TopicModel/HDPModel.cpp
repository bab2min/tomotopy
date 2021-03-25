#include "HDPModel.hpp"

namespace tomoto
{
    IHDPModel* IHDPModel::create(TermWeight _weight, const HDPArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, HDPModel, args);
	}
}