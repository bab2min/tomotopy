/*
AVX-512 implementation of log function

Based on "avx_mathfun.h" by Giovanni Garberoglio
and "sse_mathfun.h" by Julien Pommier
*/

#pragma once
#include <immintrin.h>

#ifdef _MSC_VER
#define ALIGN64_BEG __declspec(align(64))
#define ALIGN64_END
#else
#define ALIGN64_BEG
#define ALIGN64_END __attribute__((aligned(64)))
#endif

typedef __m512  v16sf;
typedef __m512i v16si;

#define _PS512_CONST(Name, Val) \
  static const ALIGN64_BEG float _ps512_##Name[16] ALIGN64_END = { Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST512(Name, Val) \
  static const ALIGN64_BEG int _pi32_512_##Name[16] ALIGN64_END = { Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS512_CONST_TYPE(Name, Type, Val) \
  static const ALIGN64_BEG Type _ps512_##Name[16] ALIGN64_END = { Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val, Val }

_PS512_CONST(1, 1.0f);
_PS512_CONST(0p5, 0.5f);
_PS512_CONST_TYPE(min_norm_pos, unsigned int, 0x00800000u);
_PS512_CONST_TYPE(inv_mant_mask, unsigned int, ~0x7f800000u);

_PI32_CONST512(0x7f, 0x7f);

_PS512_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS512_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS512_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS512_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS512_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS512_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS512_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS512_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS512_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS512_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS512_CONST(cephes_log_q1, -2.12194440e-4f);
_PS512_CONST(cephes_log_q2, 0.693359375f);

/* natural logarithm computed for 16 simultaneous float */
inline v16sf log512_ps(v16sf x) {
	v16si imm0;
	v16sf one = *(v16sf*)_ps512_1;

	__mmask16 invalid_mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LE_OS);

	x = _mm512_max_ps(x, *(v16sf*)_ps512_min_norm_pos);

	imm0 = _mm512_srli_epi32(_mm512_castps_si512(x), 23);

	x = _mm512_and_ps(x, *(v16sf*)_ps512_inv_mant_mask);
	x = _mm512_or_ps(x, *(v16sf*)_ps512_0p5);

	imm0 = _mm512_sub_epi32(imm0, *(v16si*)_pi32_512_0x7f);
	v16sf e = _mm512_cvtepi32_ps(imm0);

	e = _mm512_add_ps(e, one);

	__mmask16 mask = _mm512_cmp_ps_mask(x, *(v16sf*)_ps512_cephes_SQRTHF, _CMP_LT_OS);
	v16sf tmp = _mm512_maskz_mov_ps(mask, x);
	x = _mm512_sub_ps(x, one);
	e = _mm512_mask_sub_ps(e, mask, e, one);
	x = _mm512_add_ps(x, tmp);

	v16sf z = _mm512_mul_ps(x, x);

	v16sf y = *(v16sf*)_ps512_cephes_log_p0;
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p1);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p2);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p3);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p4);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p5);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p6);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p7);
	y = _mm512_fmadd_ps(y, x, *(v16sf*)_ps512_cephes_log_p8);
	y = _mm512_mul_ps(y, x);

	y = _mm512_mul_ps(y, z);

	y = _mm512_fmadd_ps(e, *(v16sf*)_ps512_cephes_log_q1, y);

	tmp = _mm512_mul_ps(z, *(v16sf*)_ps512_0p5);
	y = _mm512_sub_ps(y, tmp);

	x = _mm512_fmadd_ps(e, *(v16sf*)_ps512_cephes_log_q2, _mm512_add_ps(x, y));

	// negative arg will be NAN
	x = _mm512_mask_blend_ps(invalid_mask, x, _mm512_set1_ps(__builtin_nanf("")));
	return x;
}
