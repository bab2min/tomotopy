#include "LLDAModel.hpp"

namespace tomoto
{
	ILLDAModel* ILLDAModel::create(TermWeight _weight, const LDAArgs& args, bool scalarRng)
	{
		TMT_SWITCH_TW(_weight, scalarRng, LLDAModel, args);
	}
}