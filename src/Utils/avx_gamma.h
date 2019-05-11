#pragma once
#include "avx_mathfun.h"

// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
inline __m256 lgamma256_ps(__m256 x)
{
	__m256 x_3 = _mm256_add_ps(x, _mm256_set1_ps(3));
	__m256 ret = _mm256_mul_ps(_mm256_add_ps(x_3, _mm256_set1_ps(-0.5f)), log256_ps(x_3));
	ret = _mm256_sub_ps(ret, x_3);
	ret = _mm256_add_ps(ret, _mm256_set1_ps(0.91893853f));
	ret = _mm256_add_ps(ret, _mm256_div_ps(_mm256_set1_ps(1 / 12.f), x_3));
	ret = _mm256_sub_ps(ret, log256_ps(_mm256_mul_ps(
		_mm256_mul_ps(_mm256_sub_ps(x_3, _mm256_set1_ps(1)), _mm256_sub_ps(x_3, _mm256_set1_ps(2))), x)));
	return ret;
}

// approximation : digamma(z) ~= ln(z+4) - 1/2/(z+4) - 1/12/(z+4)^2 - 1/z - 1/(z+1) - 1/(z+2) - 1/(z+3)
inline __m256 digamma256_ps(__m256 x)
{
	__m256 x_4 = _mm256_add_ps(x, _mm256_set1_ps(4));
	__m256 ret = log256_ps(x_4);
	ret = _mm256_sub_ps(ret, _mm256_div_ps(_mm256_set1_ps(1 / 2.f), x_4));
	ret = _mm256_sub_ps(ret, _mm256_div_ps(_mm256_div_ps(_mm256_set1_ps(1 / 12.f), x_4), x_4));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(1))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(2))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(3))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(4))));
	return ret;
}
