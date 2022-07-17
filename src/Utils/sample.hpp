#pragma once

#include <random>
#if defined(__AVX__) || defined(__SSE2__)
#include <immintrin.h>
#else

#endif

#ifdef _WIN32
#include <intrin.h>
#endif

namespace tomoto
{
	namespace sample
	{
#ifdef _WIN32
		inline uint32_t popcnt(uint32_t i)
		{
			return __popcnt(i);
		}

	#ifdef _WIN64
		inline uint64_t log2_ceil(uint64_t i)
		{
			unsigned long idx;
			if (!_BitScanReverse64(&idx, i)) return 0;
			return idx + 1 - ((i & (i - 1)) == 0 ? 1 : 0);
		}
	#else
		inline uint32_t log2_ceil(uint32_t i)
		{
			unsigned long idx;
			if (!_BitScanReverse(&idx, i)) return 0;
			return idx + 1 - ((i & (i - 1)) == 0 ? 1 : 0);
		}
	#endif

#else
		inline uint32_t popcnt(uint32_t i)
		{
			return __builtin_popcount(i);
		}

	#ifdef __x86_64
		inline uint64_t log2_ceil(uint64_t i)
		{
			return 64 - __builtin_clzll(i) - ((i & (i - 1)) == 0 ? 1 : 0);
		}
	#else
		inline uint32_t log2_ceil(uint32_t i)
		{
			return 32 - __builtin_clz(i) - ((i & (i - 1)) == 0 ? 1 : 0);
		}
	#endif

#endif


#if defined(__SSE2__)
		inline __m128 scan_SSE(__m128 x)
		{
			x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
			x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
			return x;
		}

		inline void prefixSum(float* arr, int n)
		{
			int n4 = n & ~3;
			__m128 offset = _mm_setzero_ps();
			for (int i = 0; i < n4; i += 4)
			{
				__m128 x = _mm_load_ps(&arr[i]);
				__m128 out = scan_SSE(x);
				out = _mm_add_ps(out, offset);
				_mm_store_ps(&arr[i], out);
				offset = _mm_shuffle_ps(out, out, _MM_SHUFFLE(3, 3, 3, 3));
			}
			if (!n4) n4 = 1;
			for (int i = n4; i < n; ++i)
			{
				arr[i] += arr[i - 1];
			}
		}
#else
		inline void prefixSum(float* arr, int n)
		{
			int n4 = n & ~3;
			float acc = 0;
			for (int i = 0; i < n4; i += 4)
			{
				// first accumulation
				arr[i + 3] += arr[i + 2];
				arr[i + 2] += arr[i + 1];
				arr[i + 1] += arr[i];

				// second accumulation
				arr[i + 3] += arr[i + 1];
				arr[i + 2] += arr[i];

				// accumulate offset
				arr[i] += acc;
				arr[i + 1] += acc;
				arr[i + 2] += acc;
				arr[i + 3] += acc;

				acc = arr[i + 3];
			}

			if (!n4) n4 = 1;
			for (size_t i = n4; i < n; ++i)
			{
				arr[i] += arr[i - 1];
			}
		}
#endif

		template<class RealIt, class Random>
		inline size_t sampleFromDiscrete(RealIt begin, RealIt end, Random& rg)
		{
			auto r = rg.uniform_real() * std::accumulate(begin, end, 0.f);
			size_t K = std::distance(begin, end);
			size_t z = 0;
			for (; r > *begin && z < K - 1; ++z, ++begin)
			{
				r -= *begin;
			}
			return z;
		}

		template<class RealIt, class Random>
		inline size_t sampleFromDiscreteAcc(RealIt begin, RealIt end, Random& rg)
		{
			auto r = rg.uniform_real() * *(end - 1);
			size_t K = std::distance(begin, end);
			size_t z = 0;
#ifdef __AVX__
			__m256 mr = _mm256_set1_ps(r), mz;
			int mask;
			for (; z < (K >> 5) << 5; z += 32)
			{
				mz = _mm256_load_ps(&begin[z]);
				mask = _mm256_movemask_ps(_mm256_cmp_ps(mr, mz, _CMP_LT_OQ));
				if (mask) return z + 8 - popcnt(mask);
				mz = _mm256_load_ps(&begin[z + 8]);
				mask = _mm256_movemask_ps(_mm256_cmp_ps(mr, mz, _CMP_LT_OQ));
				if (mask) return z + 16 - popcnt(mask);
				mz = _mm256_load_ps(&begin[z + 16]);
				mask = _mm256_movemask_ps(_mm256_cmp_ps(mr, mz, _CMP_LT_OQ));
				if (mask) return z + 24 - popcnt(mask);
				mz = _mm256_load_ps(&begin[z + 24]);
				mask = _mm256_movemask_ps(_mm256_cmp_ps(mr, mz, _CMP_LT_OQ));
				if (mask) return z + 32 - popcnt(mask);
			}
			for (; z < (K >> 3) << 3; z += 8)
			{
				__m256 mz = _mm256_load_ps(&begin[z]);
				int mask = _mm256_movemask_ps(_mm256_cmp_ps(mr, mz, _CMP_LT_OQ));
				if (mask) return z + 8 - popcnt(mask);
			}
#elif defined(__SSE2__)
			__m128 mr = _mm_set1_ps(r);
			for (; z < (K >> 2) << 2; z += 4)
			{
				__m128 mz = _mm_load_ps(&begin[z]);
				int mask = _mm_movemask_ps(_mm_cmplt_ps(mr, mz));
				if (mask) return z + 4 - popcnt(mask);
			}
#else
			for (; z < (K >> 3) << 3; z += 8)
			{
				if (r < begin[z]) return z;
				if (r < begin[z + 1]) return z + 1;
				if (r < begin[z + 2]) return z + 2;
				if (r < begin[z + 3]) return z + 3;
				if (r < begin[z + 4]) return z + 4;
				if (r < begin[z + 5]) return z + 5;
				if (r < begin[z + 6]) return z + 6;
				if (r < begin[z + 7]) return z + 7;
			}
#endif
			for (; z < K; ++z)
			{
				if (r < begin[z]) return z;
			}
			return K - 1;
		}
	}
}

#include "AliasMethod.hpp"

