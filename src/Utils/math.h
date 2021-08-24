#pragma once

#include <cmath>
#include <random>
#include <cfloat>
#include "LUT.hpp"

namespace tomoto
{
	namespace math
	{
		constexpr auto pi_l = 3.1415926535897932384626433832795029L;
		constexpr auto gamma_l = 0.5772156649015328606065120900824024L;
		constexpr auto ln2_l = 0.6931471805599453094172321214581766L;
		constexpr auto pi = 3.14159265f;
		constexpr auto gamma = 0.577215664f;
		constexpr auto ln2 = 0.693147180f;

		/*
			This digamma implmentation is from http://www2.mpia-hd.mpg.de/~mathar/progs/digamma.c, and modified by bab2min.
		*/
		inline long double digammal(long double x)
		{
			if (x < 0.0L)
				return digammal(1.0L - x) + pi_l / tanl(pi_l*(1.0L - x));
			else if (x < 1.0L)
				return digammal(1.0L + x) - 1.0L / x;
			else if (x == 1.0L)
				return -gamma_l;
			else if (x == 2.0L)
				return 1.0L - gamma_l;
			else if (x == 3.0L)
				return 1.5L - gamma_l;
			else if (x > 3.0L)
				return 0.5L*(digammal(x / 2.0L) + digammal((x + 1.0L) / 2.0L)) + ln2_l;
			else
			{
				static long double Kncoe[] = { .30459198558715155634315638246624251L,
					.72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
					.27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
					.17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
					.11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
					.83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
					.59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
					.42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
					.304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
					.21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
					.15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
					.11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
					.80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
					.58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
					.41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L };

				long double Tn_1 = 1.0L;
				long double Tn = x - 2.0L;
				long double resul = Kncoe[0] + Kncoe[1] * Tn;

				x -= 2.0L;

				for (size_t n = 2; n < sizeof(Kncoe) / sizeof(long double); n++)
				{
					const long double Tn1 = 2.0L * x * Tn - Tn_1;
					resul += Kncoe[n] * Tn1;
					Tn_1 = Tn;
					Tn = Tn1;
				}
				return resul;
			}
		}

		inline float digammaf(float x)
		{
			if (x < 0.0f)
				return digammaf(1.0f - x) + pi / tanf(pi*(1.0f - x));
			else if (x < 1.0f)
				return digammaf(1.0f + x) - 1.0f / x;
			else if (x == 1.0f)
				return -gamma;
			else if (x == 2.0f)
				return 1.0L - gamma;
			else if (x == 3.0f)
				return 1.5L - gamma;
			else if (x > 3.0f)
				return 0.5f*(digammaf(x / 2.0f) + digammaf((x + 1.0f) / 2.0f)) + ln2;
			else
			{
				static float Kncoe[] = { .304591985f,
					.720379774f, -.124549592f,
					.277694573e-1f, -.677623714e-2f,
					.172387551e-2f, -.448176990e-3f,
					.117936600e-3f, -.312538942e-4f,
					.831739970e-5f, -.221914276e-5f,
					.593022667e-6f, -.158630511e-6f,
					.424592039e-7f, -.113691296e-7f,
					.304502217e-8f, -.815684550e-9f, };

				float Tn_1 = (float)1.0;
				float Tn = (float)(x - 2.0);
				float resul = Kncoe[0] + Kncoe[1] * Tn;

				x -= 2.0L;

				for (size_t n = 2; n < sizeof(Kncoe) / sizeof(float); n++)
				{
					const float Tn1 = (float)(2.0L * x * Tn - Tn_1);
					resul += Kncoe[n] * Tn1;
					Tn_1 = Tn;
					Tn = Tn1;
				}
				return resul;
			}
		}

		inline float digamma(float x) { return digammaf(x); }
		inline long double digamma(long double x) { return digammal(x); }

		namespace detail
		{
			struct F_lgamma
			{
				float operator()(float x) { return lgamma(x); }
				static constexpr float smallThreshold = (float)(0.001);
				float forSmall(float x) 
				{ 
					if (x == 0) return INFINITY; 
					return (x + 0.5f) * log(x + 1) - (x + 1) + log(2 * pi) / 2 + 1 / 12.f / (x + 1) - log(x);
				}
				float forLarge(float x) { return (x - 0.5f) * log(x) - x + log(2 * pi) / 2 + 1 / 12.f / x; }
				float forNonFinite(float x) { if (std::isnan(x)) return NAN; if (x > 0) return INFINITY; return -INFINITY; }
			};

			using LUT_lgamma = LUT3<F_lgamma, float, 1 * 1024, 1024, 100 * 64, 64, 1000 * 8, 8>;

			struct F_digamma
			{
				float operator()(float x)
				{
					return digamma(x);
				}
				static constexpr float smallThreshold = (float)(0.001);
				float forSmall(float x) 
				{
					if (x == 0) return -INFINITY;
					return logf(x + 2) - 0.5f / (x + 2) - 1 / 12.f / powf(x + 2, 2) - 1 / (x + 1) - 1 / x; 
				}
				float forLarge(float x) { return logf(x) - 0.5f / x - 1 / 12.f / powf(x, 2); }
				float forNonFinite(float x) { if (std::isnan(x) || x < 0) return NAN; return INFINITY;}
			};

			using LUT_digamma = LUT3<F_digamma, float, 1 * 1024, 1024, 100 * 64, 64, 1000 * 4, 4>;
		}

		inline float lgammaT(float x) { return detail::LUT_lgamma::get(x); }
		inline float digammaT(float x) { return detail::LUT_digamma::get(x); }

		// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
		template<class _T>
		inline auto lgammaApprox(_T z)
		{
			return (z + 2.5) * log(z + 3) - (z + 3) + 0.91893853 + 1. / 12. / (z + 3) - log(z * (z + 1) * (z + 2));
		}

		// calc lgamma(z + a) - lgamma(z)
		template<class _T, class _U>
		inline auto lgammaSubt(_T z, _U a)
		{
			return (z + a + 1.5) * log(z + a + 2) - (z + 1.5) * log(z + 2) - a + (1. / (z + a + 2) - 1. / (z + 2)) / 12. - log((z + a) / z * (z + a + 1) / (z + 1));
		}

		// approximation : digamma(z) ~= ln(z+4) - 1/2/(z+4) - 1/12/(z+4)^2 - 1/z - 1/(z+1) - 1/(z+2) - 1/(z+3)
		template<class _T>
		inline auto digammaApprox(_T z)
		{
			return log(z + 4) - 1. / 2. / (z + 4) - 1. / 12. / ((z + 4) * (z + 4)) - 1. / z - 1. / (z + 1) - 1. / (z + 2) - 1. / (z + 3);
		}

		// calc digamma(z + a) - digamma(z)
		template<class _T, class _U>
		inline auto digammaSubt(_T z, _U a)
		{
			return log((z + a + 2) / (z + 2)) - (1 / (z + a + 2) - 1 / (z + 2)) / 2 - (1 / (z + a + 2) / (z + a + 2) - 1 / (z + 2) / (z + 2)) / 12
				- 1. / (z + a) - 1. / (z + a + 1)
				- 1. / z - 1. / (z + 1);
		}

		template <typename RealType = double>
		class beta_distribution
		{
		public:
			typedef RealType result_type;

			class param_type
			{
			public:
				typedef beta_distribution distribution_type;

				explicit param_type(RealType a = 2.0, RealType b = 2.0)
					: a_param(a), b_param(b) { }

				RealType a() const { return a_param; }
				RealType b() const { return b_param; }

				bool operator==(const param_type& other) const
				{
					return (a_param == other.a_param &&
						b_param == other.b_param);
				}

				bool operator!=(const param_type& other) const
				{
					return !(*this == other);
				}

			private:
				RealType a_param, b_param;
			};

			explicit beta_distribution(RealType a = 2.0, RealType b = 2.0)
				: a_gamma(a), b_gamma(b) { }
			explicit beta_distribution(const param_type& param)
				: a_gamma(param.a()), b_gamma(param.b()) { }

			void reset() { }

			param_type param() const
			{
				return param_type(a(), b());
			}

			void param(const param_type& param)
			{
				a_gamma = gamma_dist_type(param.a());
				b_gamma = gamma_dist_type(param.b());
			}

			template <typename URNG>
			result_type operator()(URNG& engine)
			{
				return generate(engine, a_gamma, b_gamma);
			}

			template <typename URNG>
			result_type operator()(URNG& engine, const param_type& param)
			{
				gamma_dist_type a_param_gamma(param.a()),
					b_param_gamma(param.b());
				return generate(engine, a_param_gamma, b_param_gamma);
			}

			result_type min() const { return 0.0; }
			result_type max() const { return 1.0; }

			result_type a() const { return a_gamma.alpha(); }
			result_type b() const { return b_gamma.alpha(); }

			bool operator==(const beta_distribution<result_type>& other) const
			{
				return (param() == other.param() &&
					a_gamma == other.a_gamma &&
					b_gamma == other.b_gamma);
			}

			bool operator!=(const beta_distribution<result_type>& other) const
			{
				return !(*this == other);
			}

		private:
			typedef std::gamma_distribution<result_type> gamma_dist_type;

			gamma_dist_type a_gamma, b_gamma;

			template <typename URNG>
			result_type generate(URNG& engine,
				gamma_dist_type& x_gamma,
				gamma_dist_type& y_gamma)
			{
				result_type x = x_gamma(engine);
				return x / (x + y_gamma(engine));
			}
		};
	}
}