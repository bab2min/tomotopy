#include "GDMRModel.hpp"

namespace tomoto
{
    IGDMRModel* IGDMRModel::create(TermWeight _weight, const GDMRArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, GDMRModel, args);
	}
}
