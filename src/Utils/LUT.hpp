#pragma once
#include <array>

namespace tomoto
{
	namespace math
	{
		namespace detail
		{
			template<typename _Func, typename _Prec, size_t N, size_t S, size_t M, size_t T, size_t L, size_t U>
			class LUT3
			{
			protected:
				std::array<_Prec, N + M + L> points = {};
				static constexpr _Prec P = (_Prec)(1. / S);
				static constexpr _Prec Q = (_Prec)(1. / T);
				static constexpr _Prec R = (_Prec)(1. / U);
				LUT3()
				{
					_Func fun;
					for (size_t i = 0; i < N; i++)
					{
						points[i] = fun(i ? i * P : (_Prec)0.0001);
					}
					for (size_t i = 0; i < M; i++)
					{
						points[i + N] = fun(i*Q + N * P);
					}
					for (size_t i = 0; i < L; i++)
					{
						points[i + N + M] = fun(i*R + N * P + M * Q);
					}
				}

				_Prec _get(_Prec x) const
				{
					if (!std::isfinite(x)) return _Func{}.forNonFinite(x);
					if (x < 0) return NAN;
					if (x < _Func::smallThreshold) return _Func{}.forSmall(x);
					if (x >= N * P + M * Q + (L - 1) * R) return _Func{}.forLarge(x);
					size_t idx;
					_Prec a;
					_Prec nx = x;
					if (x < N*P)
					{
						idx = (size_t)(nx / P);
						a = (nx - idx * P) / P;
					}
					else
					{
						nx -= N * P;
						if (nx < M*Q)
						{
							idx = (size_t)(nx / Q);
							a = (nx - idx * Q) / Q;
							idx += N;
						}
						else
						{
							nx -= M * Q;
							idx = (size_t)(nx / R);
							a = (nx - idx * R) / R;
							idx += N + M;
						}
					}
					return points[idx] + a * (points[idx + 1] - points[idx]);
				}
			public:
				static const LUT3& getInst()
				{
					static LUT3 lg;
					return lg;
				}

				static _Prec get(_Prec x)
				{
					return getInst()._get(x);
				}
			};
		}
	}
}