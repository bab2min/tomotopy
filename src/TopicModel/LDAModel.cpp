#include "LDAModel.hpp"

namespace tomoto
{
	template class LDAModel<TermWeight::one>;
	template class LDAModel<TermWeight::idf>;
	template class LDAModel<TermWeight::pmi>;

	template<TermWeight _TW, size_t _Flags,
		typename _Interface,
		typename _Derived,
		typename _DocType,
		typename _ModelState>
	constexpr const char* LDAModel<_TW, _Flags, _Interface, _Derived, _DocType, _ModelState>::TWID;
	
	template<TermWeight _TW, size_t _Flags,
		typename _Interface,
		typename _Derived,
		typename _DocType,
		typename _ModelState>
	constexpr const char LDAModel<_TW, _Flags, _Interface, _Derived, _DocType, _ModelState>::TMID[5];

    ILDAModel* ILDAModel::create(TermWeight _weight, size_t _K, Float _alpha, Float _eta, const RandGen& _rg)
    {
        SWITCH_TW(_weight, LDAModel, _K, _alpha, _eta, _rg);
    }
}
