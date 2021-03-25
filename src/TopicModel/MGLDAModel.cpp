#include "MGLDAModel.hpp"

namespace tomoto
{
    IMGLDAModel* IMGLDAModel::create(TermWeight _weight, const MGLDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, MGLDAModel, args);
	}
}