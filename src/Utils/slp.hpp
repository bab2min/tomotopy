#pragma once
#include <cstdint>

namespace tomoto
{
	namespace slp
	{
		template<int n, int k>
		struct combination
		{
			static constexpr int64_t value = combination<n - 1, k - 1>::value + combination<n - 1, k>::value;
		};

		template<int n>
		struct combination<n, 0>
		{
			static constexpr int64_t value = 1;
		};

		template<int n>
		struct combination<0, n>
		{
			static constexpr int64_t value = 1;
		};

		template<int n>
		struct combination<n, n>
		{
			static constexpr int64_t value = 1;
		};

		template<>
		struct combination<0, 0>
		{
			static constexpr int64_t value = 1;
		};

		template<int _Order, class _Type, int n = 0>
		struct shiftedLegendre
		{
			inline static _Type at(_Type x)
			{
				return shiftedLegendre<_Order, _Type, n + 1>::at(x) * x + combination<_Order, n>::value * combination<_Order + n, n>::value * ((_Order + n) % 2 ? -1 : 1);
			}

			inline static _Type atDerivative(_Type x)
			{
				return shiftedLegendre<_Order, _Type, n + 1>::atDerivation(x) * x + combination<_Order, n>::value * combination<_Order + n, n>::value * ((_Order + n) % 2 ? -1 : 1) * (int)n;
			}
		};

		template<int _Order, class _Type>
		struct shiftedLegendre<_Order, _Type, _Order>
		{
			inline static _Type at(_Type x)
			{
				return combination<_Order + _Order, _Order>::value;
			}

			inline static _Type atDerivative(_Type x)
			{
				return combination<_Order + _Order, _Order>::value * _Order;
			}
		};

		template<class _Type>
		struct shiftedLegendre<0, _Type, 0>
		{
			inline static _Type at(_Type x)
			{
				return 1;
			}
		};

		template<class _Type>
		struct shiftedLegendre<0, _Type, 1>
		{
			inline static _Type atDerivative(_Type x)
			{
				return 0;
			}
		};

		template<int _Order, class _Type> inline _Type shiftedLegendreFunc(_Type x)
		{
			return shiftedLegendre<_Order, _Type, 0>::at(x);
		}

		template<int _Order, class _Type> inline _Type shiftedLegendreFuncDerivative(_Type x)
		{
			return shiftedLegendre<_Order, _Type, 1>::atDerivative(x);
		}


		template<class _Type> _Type slpGet(size_t order, _Type x)
		{
			switch (order)
			{
			case 0: return shiftedLegendreFunc<0>(x);
			case 1: return shiftedLegendreFunc<1>(x);
			case 2: return shiftedLegendreFunc<2>(x);
			case 3: return shiftedLegendreFunc<3>(x);
			case 4: return shiftedLegendreFunc<4>(x);
			case 5: return shiftedLegendreFunc<5>(x);
			case 6: return shiftedLegendreFunc<6>(x);
			case 7: return shiftedLegendreFunc<7>(x);
			case 8: return shiftedLegendreFunc<8>(x);
			case 9: return shiftedLegendreFunc<9>(x);
			case 10: return shiftedLegendreFunc<10>(x);
			case 11: return shiftedLegendreFunc<11>(x);
			case 12: return shiftedLegendreFunc<12>(x);
			case 13: return shiftedLegendreFunc<13>(x);
			case 14: return shiftedLegendreFunc<14>(x);
			case 15: return shiftedLegendreFunc<15>(x);
			}
			return _Type{};
		}

		inline size_t partialProductDown(size_t n, size_t k)
		{
			size_t ret = 1;
			for (size_t i = 0; i < k; ++i) ret *= n--;
			return ret;
		}

		inline int slpGetCoef(size_t n, size_t k)
		{
			return ((n + k) & 1 ? -1 : 1) * (int)(partialProductDown(n, k) / partialProductDown(k, k) * partialProductDown(n + k, k) / partialProductDown(k, k));
		}
	}
}