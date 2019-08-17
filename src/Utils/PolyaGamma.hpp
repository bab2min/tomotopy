#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

namespace tomoto
{
	namespace math
	{
		template<class _Real, class _RNG>
		class PolyaGamma
		{
			static constexpr _Real __PI = 3.141592653589793238462643383279502884197;
			static constexpr _Real HALFPISQ = 0.5 * __PI * __PI;
			static constexpr _Real FOURPISQ = 4 * __PI * __PI;
			static constexpr _Real __TRUNC = 0.64;
			static constexpr _Real __TRUNC_RECIP = 1.0 / __TRUNC;

			static _Real p_norm(_Real x)
			{
				return std::erf(x / std::sqrt((_Real)2)) / 2 + 0.5f;
			}

			static _Real draw_like_devroye(_Real Z, _RNG& r)
			{
				// Change the parameter.
				Z = std::fabs(Z) * 0.5;

				// Now sample 0.25 * J^*(1, Z := Z/2).
				_Real fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
				// ... Problems with large Z?  Try using q_over_p.
				// double p  = 0.5 * __PI * exp(-1.0 * fz * __TRUNC) / fz;
				// double q  = 2 * exp(-1.0 * Z) * pigauss(__TRUNC, Z);

				_Real X = 0.0;
				_Real S = 1.0;
				_Real Y = 0.0;
				// int iter = 0; If you want to keep track of iterations.

				while (true) 
				{

					// if (r.unif() < p/(p+q))
					if (std::generate_canonical<_Real, sizeof(_Real) * 8>(r) < mass_texpon(Z))
						X = __TRUNC + std::exponential_distribution<_Real>()(r) / fz;
					else
						X = rtigauss(Z, r);

					S = a(0, X);
					Y = std::generate_canonical<_Real, sizeof(_Real) * 8>(r) * S;
					int n = 0;
					bool go = true;

					// Cap the number of iterations?
					while (go) 
					{
						++n;
						if (n % 2 == 1) 
						{
							S = S - a(n, X);
							if (Y <= S) return 0.25 * X;
						}
						else 
						{
							S = S + a(n, X);
							if (Y > S) go = false;
						}

					}
					// Need Y <= S in event that Y = S, e.g. when X = 0.

				}
			}

			static _Real a(int n, _Real x)
			{
				_Real K = (n + 0.5) * __PI;
				_Real y = 0;
				if (x > __TRUNC) {
					y = K * std::exp(-0.5 * K*K * x);
				}
				else if (x > 0) {
					_Real expnt = -1.5 * (std::log(0.5 * __PI) + std::log(x)) + std::log(K) - 2.0 * (n + 0.5)*(n + 0.5) / x;
					y = std::exp(expnt);
					// y = pow(0.5 * __PI * x, -1.5) * K * exp( -2.0 * (n+0.5)*(n+0.5) / x);
					// ^- unstable for small x?
				}
				return y;
			}

			static _Real mass_texpon(_Real Z)
			{
				_Real t = __TRUNC;

				_Real fz = 0.125 * __PI*__PI + 0.5 * Z*Z;
				_Real b = std::sqrt(1.0 / t) * (t * Z - 1);
				_Real a = std::sqrt(1.0 / t) * (t * Z + 1) * -1.0;

				_Real x0 = log(fz) + fz * t;
				_Real xb = x0 - Z + log(p_norm(b));
				_Real xa = x0 + Z + log(p_norm(a));

				_Real qdivp = 4 / __PI * (exp(xb) + exp(xa));

				return 1.0 / (1.0 + qdivp);
			}

			static _Real rtigauss(_Real Z, _RNG& r)
			{
				Z = std::fabs(Z);
				_Real t = __TRUNC;
				_Real X = t + 1.0;
				if (__TRUNC_RECIP > Z) 
				{ // mu > t
					_Real alpha = 0.0;
					while (std::generate_canonical<_Real, sizeof(_Real) * 8>(r) > alpha)
					{
						// X = t + 1.0;
						// while (X > t)
						// 	X = 1.0 / r.gamma_rate(0.5, 0.5);
						// Slightly faster to use truncated normal.
						_Real E1 = std::exponential_distribution<_Real>()(r);
						_Real E2 = std::exponential_distribution<_Real>()(r);
						while (E1*E1 > 2 * E2 / t) 
						{
							E1 = std::exponential_distribution<_Real>()(r);
							E2 = std::exponential_distribution<_Real>()(r);
						}
						X = 1 + E1 * t;
						X = t / (X * X);
						alpha = std::exp(-0.5 * Z*Z * X);
					}
				}
				else 
				{
					_Real mu = 1.0 / Z;
					while (X > t)
					{
						_Real Y = std::normal_distribution<_Real>()(r); Y *= Y;
						_Real half_mu = 0.5 * mu;
						_Real mu_Y = mu * Y;
						X = mu + half_mu * mu_Y - half_mu * sqrt(4 * mu_Y + mu_Y * mu_Y);
						if (std::generate_canonical<_Real, sizeof(_Real) * 8>(r) > mu / (mu + X))
							X = mu * mu / X;
					}
				}
				return X;
			}

			static _Real jj_m1(_Real b, _Real z)
			{
				z = std::fabs(z);
				_Real m1 = 0.0;
				if (z > 1e-12)
					m1 = b * std::tanh(z) / z;
				else
					m1 = b * (1 - (1.0 / 3) * std::pow(z, 2) + (2.0 / 15) * std::pow(z, 4) - (17.0 / 315) * std::pow(z, 6));
				return m1;
			}

			static _Real jj_m2(_Real b, _Real z)
			{
				z = std::fabs(z);
				double m2 = 0.0;
				if (z > 1e-12)
					m2 = (b + 1) * b * std::pow(tanh(z) / z, 2) + b * ((std::tanh(z) - z) / std::pow(z, 3));
				else
					m2 = (b + 1) * b * std::pow(1 - (1.0 / 3) * std::pow(z, 2) + (2.0 / 15) * std::pow(z, 4) - (17.0 / 315) * std::pow(z, 6), 2) +
					b * ((-1.0 / 3) + (2.0 / 15) * std::pow(z, 2) - (17.0 / 315) * std::pow(z, 4));
				return m2;
			}

		public:
			static _Real draw(size_t n, _Real z, _RNG& r)
			{
				_Real sum = 0.0;
				for (size_t i = 0; i < n; ++i)
					sum += draw_like_devroye(z, r);
				return sum;
			}

			static _Real pg_m1(_Real b, _Real z)
			{
				return jj_m1(b, 0.5 * z) * 0.25;
			}

			static _Real pg_m2(_Real b, _Real z)
			{
				return jj_m2(b, 0.5 * z) * 0.0625;
			}

		};

		template<class _Real, class _RNG> _Real drawPolyaGamma(size_t n, _Real z, _RNG& r)
		{
			return PolyaGamma<_Real, _RNG>::draw(n, z, r);
		}
	}
}