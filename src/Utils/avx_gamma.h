#pragma once
#include "avx_mathfun.h"

// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
inline __m256 lgamma_ps(__m256 x)
{
	__m256 x_3 = _mm256_add_ps(x, _mm256_set1_ps(3));
	__m256 ret = _mm256_mul_ps(_mm256_add_ps(x_3, _mm256_set1_ps(-0.5f)), log_ps(x_3));
	ret = _mm256_sub_ps(ret, x_3);
	ret = _mm256_add_ps(ret, _mm256_set1_ps(0.91893853f));
	ret = _mm256_add_ps(ret, _mm256_div_ps(_mm256_set1_ps(1 / 12.f), x_3));
	ret = _mm256_sub_ps(ret, log_ps(_mm256_mul_ps(
		_mm256_mul_ps(_mm256_sub_ps(x_3, _mm256_set1_ps(1)), _mm256_sub_ps(x_3, _mm256_set1_ps(2))), x)));
	return ret;
}

// approximation : lgamma(z + a) - lgamma(z) = (z + a + 1.5) * log(z + a + 2) - (z + 1.5) * log(z + 2) - a + (1. / (z + a + 2) - 1. / (z + 2)) / 12. - log(((z + a) * (z + a + 1)) / (z * (z + 1)))
inline __m256 lgamma_subt(__m256 z, __m256 a)
{
	__m256 _1p5 = _mm256_set1_ps(1.5);
	__m256 _2 = _mm256_set1_ps(2);
	__m256 za = _mm256_add_ps(z, a);
	__m256 ret = _mm256_mul_ps(_mm256_add_ps(za, _1p5), log_ps(_mm256_add_ps(za, _2)));
	ret = _mm256_sub_ps(ret, _mm256_mul_ps(_mm256_add_ps(z, _1p5), log_ps(_mm256_add_ps(z, _2))));
	ret = _mm256_sub_ps(ret, a);
	__m256 _1 = _mm256_set1_ps(1);
	__m256 _1_12 = _mm256_set1_ps(1 / 12.f);
	ret = _mm256_add_ps(ret, _mm256_sub_ps(_mm256_div_ps(_1_12, _mm256_add_ps(za, _2)), _mm256_div_ps(_1_12, _mm256_add_ps(z, _2))));
	ret = _mm256_sub_ps(ret, log_ps(_mm256_div_ps(_mm256_div_ps(_mm256_mul_ps(za, _mm256_add_ps(za, _1)), z), _mm256_add_ps(z, _1))));
	return ret;
}


// approximation : digamma(z) ~= ln(z+4) - 1/2/(z+4) - 1/12/(z+4)^2 - 1/z - 1/(z+1) - 1/(z+2) - 1/(z+3)
inline __m256 digamma_ps(__m256 x)
{
	__m256 x_4 = _mm256_add_ps(x, _mm256_set1_ps(4));
	__m256 ret = log_ps(x_4);
	ret = _mm256_sub_ps(ret, _mm256_div_ps(_mm256_set1_ps(1 / 2.f), x_4));
	ret = _mm256_sub_ps(ret, _mm256_div_ps(_mm256_div_ps(_mm256_set1_ps(1 / 12.f), x_4), x_4));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(1))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(2))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(3))));
	ret = _mm256_sub_ps(ret, _mm256_rcp_ps(_mm256_sub_ps(x_4, _mm256_set1_ps(4))));
	return ret;
}
