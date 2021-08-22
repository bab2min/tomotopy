#pragma once
#include <memory>
#include <iterator>
#include <limits>

namespace tomoto
{
	namespace sample
	{
		template<typename _Precision = uint32_t>
		class AliasMethod
		{
			std::unique_ptr<_Precision[]> arr;
			std::unique_ptr<size_t[]> alias;
			size_t msize = 0, bitsize = 0;

		public:
			AliasMethod()
			{
			}

			AliasMethod(const AliasMethod& o)
			{
				operator=(o);
			}

			AliasMethod(AliasMethod&& o)
			{
				operator=(o);
			}

			AliasMethod& operator=(const AliasMethod& o)
			{
				msize = o.msize;
				bitsize = o.bitsize;
				if (msize)
				{
					size_t n = (size_t)1 << bitsize;
					arr = std::make_unique<_Precision[]>(n);
					alias = std::make_unique<size_t[]>(n);

					std::copy(o.arr.get(), o.arr.get() + n, arr.get());
					std::copy(o.alias.get(), o.alias.get() + n, alias.get());
				}
				return *this;
			}

			AliasMethod& operator=(AliasMethod&& o)
			{
				msize = o.msize;
				bitsize = o.bitsize;
				std::swap(arr, o.arr);
				std::swap(alias, o.alias);
				return *this;
			}

			template<typename _Iter>
			AliasMethod(_Iter first, _Iter last)
			{
				buildTable(first, last);
			}

			template<typename _Iter>
			void buildTable(_Iter first, _Iter last)
			{
				size_t psize, nbsize;
				msize = 0;
				double sum = 0;
				for (auto it = first; it != last; ++it, ++msize)
				{
					sum += *it;
				}

				if (!std::isfinite(sum)) THROW_ERROR_WITH_INFO(exc::InvalidArgument, "cannot build NaN value distribution");

				// ceil to power of 2
				nbsize = log2_ceil(msize);
				psize = (size_t)1 << nbsize;

				if (nbsize != bitsize)
				{
					arr = std::make_unique<_Precision[]>(psize);
					std::fill(arr.get(), arr.get() + psize, 0);
					alias = std::make_unique<size_t[]>(psize);
					bitsize = nbsize;
				}
				
				sum /= psize;

				auto f = std::make_unique<double[]>(psize);
				auto pf = f.get();
				for (auto it = first; it != last; ++it, ++pf)
				{
					*pf = *it / sum;
				}
				std::fill(pf, pf + psize - msize, 0);

				size_t over = 0, under = 0, mm;
				while (over < psize && f[over] < 1) ++over;
				while (under < psize && f[under] >= 1) ++under;
				mm = under + 1;

				while (over < psize && under < psize)
				{
					arr[under] = f[under] * (std::numeric_limits<_Precision>::max() + 1.0);
					alias[under] = over;
					f[over] += f[under] - 1;
					if (f[over] >= 1 || mm <= over)
					{
						for (under = mm; under < psize && f[under] >= 1; ++under);
						mm = under + 1;
					}
					else
					{
						under = over;
					}

					while (over < psize && f[over] < 1) ++over;
				}

				for (; over < psize; ++over)
				{
					if (f[over] >= 1)
					{
						arr[over] = std::numeric_limits<_Precision>::max();
						alias[over] = over;
					}
				}

				if (under < psize)
				{
					arr[under] = std::numeric_limits<_Precision>::max();
					alias[under] = under;
					for (under = mm; under < msize; ++under)
					{
						if (f[under] < 1)
						{
							arr[under] = std::numeric_limits<_Precision>::max();
							alias[under] = under;
						}
					}
				}
			}

			template<typename _Rng>
			size_t operator()(_Rng& rng) const
			{
				auto x = rng();
				size_t a;
				if (sizeof(_Precision) < sizeof(typename _Rng::result_type))
				{
					a = x >> (sizeof(x) * 8 - bitsize);
				}
				else
				{
					a = rng() & ((1 << bitsize) - 1);
				}
				
				_Precision b = (_Precision)x;
				if (b < arr[a])
				{
					assert(a < msize);
					return a;
				}
				assert(alias[a] < msize);
				return alias[a];
			}
		};
	}
}
