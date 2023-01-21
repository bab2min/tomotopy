#pragma once

inline float32x4_t accurate_rcp(float32x4_t x)
{
	float32x4_t r = vrecpeq_f32(x);
	return vmulq_f32(vrecpsq_f32(x, r), r);
}

// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
inline float32x4_t lgamma_ps(float32x4_t x)
{
	float32x4_t x_3 = vaddq_f32(x, vmovq_n_f32(3));
	float32x4_t ret = vmulq_f32(vaddq_f32(x_3, vmovq_n_f32(-0.5f)), Eigen::internal::plog(x_3));
	ret = vsubq_f32(ret, x_3);
	ret = vaddq_f32(ret, vmovq_n_f32(0.91893853f));
	ret = vaddq_f32(ret, vdivq_f32(vmovq_n_f32(1 / 12.f), x_3));
	ret = vsubq_f32(ret, Eigen::internal::plog(vmulq_f32(
		vmulq_f32(vsubq_f32(x_3, vmovq_n_f32(1)), vsubq_f32(x_3, vmovq_n_f32(2))), x)));
	return ret;
}

// approximation : lgamma(z + a) - lgamma(z) = (z + a + 1.5) * log(z + a + 2) - (z + 1.5) * log(z + 2) - a + (1. / (z + a + 2) - 1. / (z + 2)) / 12. - log(((z + a) * (z + a + 1)) / (z * (z + 1)))
inline float32x4_t lgamma_subt(float32x4_t z, float32x4_t a)
{
	float32x4_t _1p5 = vmovq_n_f32(1.5);
	float32x4_t _2 = vmovq_n_f32(2);
	float32x4_t za = vaddq_f32(z, a);
	float32x4_t ret = vmulq_f32(vaddq_f32(za, _1p5), Eigen::internal::plog(vaddq_f32(za, _2)));
	ret = vsubq_f32(ret, vmulq_f32(vaddq_f32(z, _1p5), Eigen::internal::plog(vaddq_f32(z, _2))));
	ret = vsubq_f32(ret, a);
	float32x4_t _1 = vmovq_n_f32(1);
	float32x4_t _1_12 = vmovq_n_f32(1 / 12.f);
	ret = vaddq_f32(ret, vsubq_f32(vdivq_f32(_1_12, vaddq_f32(za, _2)), vdivq_f32(_1_12, vaddq_f32(z, _2))));
	ret = vsubq_f32(ret, Eigen::internal::plog(vdivq_f32(vdivq_f32(vmulq_f32(za, vaddq_f32(za, _1)), z), vaddq_f32(z, _1))));
	return ret;
}


// approximation : digamma(z) ~= ln(z+4) - 1/2/(z+4) - 1/12/(z+4)^2 - 1/z - 1/(z+1) - 1/(z+2) - 1/(z+3)
inline float32x4_t digamma_ps(float32x4_t x)
{
	float32x4_t x_4 = vaddq_f32(x, vmovq_n_f32(4));
	float32x4_t ret = Eigen::internal::plog(x_4);
	ret = vsubq_f32(ret, vdivq_f32(vmovq_n_f32(1 / 2.f), x_4));
	ret = vsubq_f32(ret, vdivq_f32(vdivq_f32(vmovq_n_f32(1 / 12.f), x_4), x_4));
	ret = vsubq_f32(ret, accurate_rcp(vsubq_f32(x_4, vmovq_n_f32(1))));
	ret = vsubq_f32(ret, accurate_rcp(vsubq_f32(x_4, vmovq_n_f32(2))));
	ret = vsubq_f32(ret, accurate_rcp(vsubq_f32(x_4, vmovq_n_f32(3))));
	ret = vsubq_f32(ret, accurate_rcp(vsubq_f32(x_4, vmovq_n_f32(4))));
	return ret;
}
