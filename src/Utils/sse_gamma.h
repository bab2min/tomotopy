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

// approximation : lgamma(z + a) - lgamma(z) = (z + a + 1.5) * log(z + a + 2) - (z + 1.5) * log(z + 2) - a + (1. / (z + a + 2) - 1. / (z + 2)) / 12. - log(((z + a) * (z + a + 1)) / (z * (z + 1)))
inline __m128 lgamma_subt(__m128 z, __m128 a)
{
	__m128 _1 = _mm_set1_ps(1);
	__m128 _1p5 = _mm_set1_ps(1.5);
	__m128 _2 = _mm_set1_ps(2);
	__m128 _1_12 = _mm_set1_ps(1 / 12.f);
	__m128 za = _mm_add_ps(z, a);
	__m128 ret = _mm_mul_ps(_mm_add_ps(za, _1p5), log_ps(_mm_add_ps(za, _2)));
	ret = _mm_sub_ps(ret, _mm_mul_ps(_mm_add_ps(z, _1p5), log_ps(_mm_add_ps(z, _2))));
	ret = _mm_sub_ps(ret, a);
	ret = _mm_add_ps(ret, _mm_sub_ps(_mm_div_ps(_1_12, _mm_add_ps(za, _2)), _mm_div_ps(_1_12, _mm_add_ps(z, _2))));
	ret = _mm_sub_ps(ret, log_ps(_mm_div_ps(_mm_div_ps(_mm_mul_ps(za, _mm_add_ps(za, _1)), z), _mm_add_ps(z, _1))));
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
