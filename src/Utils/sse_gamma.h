#pragma once
#include "sse_mathfun.h"

// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
inline __m128 lgamma_ps(__m128 x)
{
	__m128 x_3 = _mm_add_ps(x, _mm_set1_ps(3));
	__m128 ret = _mm_mul_ps(_mm_add_ps(x_3, _mm_set1_ps(-0.5f)), log_ps(x_3));
	ret = _mm_sub_ps(ret, x_3);
	ret = _mm_add_ps(ret, _mm_set1_ps(0.91893853f));
	ret = _mm_add_ps(ret, _mm_div_ps(_mm_set1_ps(1 / 12.f), x_3));
	ret = _mm_sub_ps(ret, log_ps(_mm_mul_ps(
		_mm_mul_ps(_mm_sub_ps(x_3, _mm_set1_ps(1)), _mm_sub_ps(x_3, _mm_set1_ps(2))), x)));
	return ret;
}

// approximation : digamma(z) ~= ln(z+4) - 1/2/(z+4) - 1/12/(z+4)^2 - 1/z - 1/(z+1) - 1/(z+2) - 1/(z+3)
inline __m128 digamma_ps(__m128 x)
{
	__m128 x_4 = _mm_add_ps(x, _mm_set1_ps(4));
	__m128 ret = log_ps(x_4);
	ret = _mm_sub_ps(ret, _mm_div_ps(_mm_set1_ps(1 / 2.f), x_4));
	ret = _mm_sub_ps(ret, _mm_div_ps(_mm_div_ps(_mm_set1_ps(1 / 12.f), x_4), x_4));
	ret = _mm_sub_ps(ret, _mm_rcp_ps(_mm_sub_ps(x_4, _mm_set1_ps(1))));
	ret = _mm_sub_ps(ret, _mm_rcp_ps(_mm_sub_ps(x_4, _mm_set1_ps(2))));
	ret = _mm_sub_ps(ret, _mm_rcp_ps(_mm_sub_ps(x_4, _mm_set1_ps(3))));
	ret = _mm_sub_ps(ret, _mm_rcp_ps(_mm_sub_ps(x_4, _mm_set1_ps(4))));
	return ret;
}
