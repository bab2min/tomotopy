#pragma once

#include <random>
#ifdef __AVX__
#include <immintrin.h>
#elif defined(__SSE2__)
#include <xmmintrin.h>
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
#else
		inline uint32_t popcnt(uint32_t i)
		{
			return __builtin_popcount(i);
		}
#endif


#ifdef __AVX__
		inline __m256 scan_AVX(__m256 x)
		{
			__m256 t0, t1;
			//shift1_AVX + add
			t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
			t1 = _mm256_permute2f128_ps(t0, t0, 41);
			x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
			//shift2_AVX + add
			t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
			t1 = _mm256_permute2f128_ps(t0, t0, 41);
			x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
			//shift3_AVX + add
			x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 41));
			return x;
		}

		inline void prefix_sum_AVX(float *a, const int n)
		{
			__m256 offset = _mm256_setzero_ps();
			for (int i = 0; i < n; i += 8)
			{
				__m256 x = _mm256_loadu_ps(&a[i]);
				__m256 out = scan_AVX(x);
				out = _mm256_add_ps(out, offset);
				_mm256_storeu_ps(&a[i], out);
				//broadcast last element
				__m256 t0 = _mm256_permute2f128_ps(out, out, 0x11);
				offset = _mm256_permute_ps(t0, 0xff);
			}
		}

		inline void prefixSum(float* arr, size_t K)
		{
			size_t Kf = (K >> 3) << 3;
			if (Kf) prefix_sum_AVX(arr, Kf);
			else Kf = 1;
			for (size_t i = Kf; i < K; ++i)
			{
				arr[i] += arr[i - 1];
			}
		}

#elif defined(__SSE2__)
		inline __m128 scan_SSE(__m128 x)
		{
			x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
			x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
			return x;
		}

		inline void prefix_sum_SSE(float *a, const int n)
		{
			__m128 offset = _mm_setzero_ps();
			for (int i = 0; i < n; i += 4)
			{
				__m128 x = _mm_load_ps(&a[i]);
				__m128 out = scan_SSE(x);
				out = _mm_add_ps(out, offset);
				_mm_store_ps(&a[i], out);
				offset = _mm_shuffle_ps(out, out, _MM_SHUFFLE(3, 3, 3, 3));
			}
		}

		inline void prefixSum(float* arr, size_t K)
		{
			size_t Kf = (K >> 2) << 2;
			if (Kf) prefix_sum_SSE(arr, Kf);
			else Kf = 1;
			for (size_t i = Kf; i < K; ++i)
			{
				arr[i] += arr[i - 1];
			}
		}
#else
		inline void prefixSum(float* arr, size_t K)
		{
			for (size_t i = 1; i < K; ++i)
			{
				arr[i] += arr[i - 1];
			}
		}
#endif
		struct FastRealGenerator
		{
			template<class Random>
			float operator()(Random& rg)
			{
				union
				{
					float f;
					uint32_t u;
				};

				u = rg();
				u = (127 << 23) | (u & 0x7FFFFF);
				return f - 1;
			}
		};

		template<class RealIt, class Random>
		inline size_t sampleFromDiscrete(RealIt begin, RealIt end, Random& rg)
		{
			FastRealGenerator dist;
			auto r = dist(rg) * std::accumulate(begin, end, 0.f);
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
			//auto r = std::generate_canonical<float, 32>(rg) * *(end - 1);
			FastRealGenerator dist;
			auto r = dist(rg) * *(end - 1);
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
