#pragma once
#include <type_traits>
#include <Eigen/Dense>
#include "math.h"

namespace Eigen
{
	namespace internal
	{
		template<typename PacketType>
		struct to_int_packet
		{
			typedef PacketType type;
		};

		inline float bit_to_ur(uint32_t x)
		{
			union
			{
				float f;
				uint32_t u;
			};

			u = x;
			u = (127 << 23) | (u & 0x7FFFFF);
			return f - 1;
		}

		template<typename PacketType, typename Rng>
		struct box_muller
		{
		};
	}
}

#ifdef __GNUC__
#if __GNUC__ < 8
#define _mm256_set_m128i(v0, v1)  _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif
#endif

#ifdef EIGEN_VECTORIZE_AVX
#include <immintrin.h>
#include "avx_gamma.h"

namespace Eigen
{
	namespace internal
	{
		template<> struct to_int_packet<Packet8f>
		{
			typedef Packet8i type;
		};

		EIGEN_STRONG_INLINE Packet8f p_to_f32(const Packet8i& a)
		{
			return _mm256_cvtepi32_ps(a);
		}

#ifdef EIGEN_VECTORIZE_AVX2
		inline Packet8f bit_to_ur(const Packet8i& x)
		{
			const __m256i lower = _mm256_set1_epi32(0x7FFFFF),
				upper = _mm256_set1_epi32(127 << 23);
			const __m256 one = _mm256_set1_ps(1);
			union
			{
				__m256 f;
				__m256i u;
			};
			u = _mm256_or_si256(_mm256_and_si256(x, lower), upper);
			return _mm256_sub_ps(f, one);
		}
#else
		inline Packet8f bit_to_ur(const Packet8i& x)
		{
			const __m128i lower = _mm_set1_epi32(0x7FFFFF),
				upper = _mm_set1_epi32(127 << 23);
			const __m256 one = _mm256_set1_ps(1);
			union
			{
				__m256 f;
				__m256i u;
			};
			u = _mm256_set_m128i(
				_mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(x, 1), lower), upper),
				_mm_or_si128(_mm_and_si128(_mm256_extractf128_si256(x, 0), lower), upper)
			);
			return _mm256_sub_ps(f, one);
		}
#endif

		template<typename Rng>
		struct box_muller<Packet8f, Rng>
		{
			Packet8f operator()(Rng&& rng, Packet8f& cache)
			{
				__m256 u1, u2;
				if (sizeof(decltype(rng())) == 8)
				{
					u1 = bit_to_ur(_mm256_set_epi64x(rng(), rng(), rng(), rng()));
					u2 = bit_to_ur(_mm256_set_epi64x(rng(), rng(), rng(), rng()));
				}
				else
				{
					u1 = bit_to_ur(_mm256_set_epi32(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()));
					u2 = bit_to_ur(_mm256_set_epi32(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()));
				}

				const __m256 twopi = _mm256_set1_ps(2.0f * 3.14159265358979323846f);
				const __m256 one = _mm256_set1_ps(1.0f);
				const __m256 minustwo = _mm256_set1_ps(-2.0f);

				u1 = _mm256_sub_ps(one, u1);

				__m256 radius = _mm256_sqrt_ps(_mm256_mul_ps(minustwo, log_ps(u1)));
				__m256 theta = _mm256_mul_ps(twopi, u2);
				__m256 sintheta, costheta;
				sincos_ps(theta, &sintheta, &costheta);
				cache = _mm256_mul_ps(radius, costheta);
				return _mm256_mul_ps(radius, sintheta);
			}
		};
	}
}

#elif defined(EIGEN_VECTORIZE_SSE2)
#include <xmmintrin.h>
#include "sse_gamma.h"

namespace Eigen
{
	namespace internal
	{
		template<> struct to_int_packet<Packet4f>
		{
			typedef Packet4i type;
		};

		EIGEN_STRONG_INLINE Packet4f p_to_f32(const Packet4i& a)
		{
			return _mm_cvtepi32_ps(a);
		}

		inline Packet4f bit_to_ur(const Packet4i& x)
		{
			const __m128i lower = _mm_set1_epi32(0x7FFFFF),
				upper = _mm_set1_epi32(127 << 23);
			const __m128 one = _mm_set1_ps(1);
			union
			{
				__m128 f;
				__m128i u;
			};
			u = _mm_or_si128(_mm_and_si128(x, lower), upper);
			return _mm_sub_ps(f, one);
		}

		template<typename Rng>
		struct box_muller<Packet4f, Rng>
		{
			Packet4f operator()(Rng&& rng, Packet4f& cache)
			{
				__m128 u1, u2;
				if (sizeof(decltype(rng())) == 8)
				{
					u1 = bit_to_ur(_mm_set_epi64x(rng(), rng()));
					u2 = bit_to_ur(_mm_set_epi64x(rng(), rng()));
				}
				else
				{
					u1 = bit_to_ur(_mm_set_epi32(rng(), rng(), rng(), rng()));
					u2 = bit_to_ur(_mm_set_epi32(rng(), rng(), rng(), rng()));
				}

				const __m128 twopi = _mm_set1_ps(2.0f * 3.14159265358979323846f);
				const __m128 one = _mm_set1_ps(1.0f);
				const __m128 minustwo = _mm_set1_ps(-2.0f);

				u1 = _mm_sub_ps(one, u1);

				__m128 radius = _mm_sqrt_ps(_mm_mul_ps(minustwo, log_ps(u1)));
				__m128 theta = _mm_mul_ps(twopi, u2);
				__m128 sintheta, costheta;
				sincos_ps(theta, &sintheta, &costheta);
				cache = _mm_mul_ps(radius, costheta);
				return _mm_mul_ps(radius, sintheta);
			}
		};

	}
}
#endif

namespace Eigen
{
	namespace internal
	{
		template<typename Scalar, typename Scalar2> struct scalar_lgamma_subt_op {
			EIGEN_EMPTY_STRUCT_CTOR(scalar_lgamma_subt_op)
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() (const Scalar& z, const Scalar2& a) const { return tomoto::math::lgammaSubt(z, a); }
			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& z, const Packet& a) const
			{
				return lgamma_subt(z, a);
			}
		};

		template<typename Scalar, typename Scalar2>
		struct functor_traits<scalar_lgamma_subt_op<Scalar, Scalar2> >
		{
			enum {
				Cost = HugeCost,
				PacketAccess = 1
			};
		};

		template<>
		struct scalar_cast_op<int32_t, float> {
			EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
				typedef float result_type;
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const float operator() (const int32_t& a) const { return cast<int32_t, float>(a); }
			
			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const typename to_int_packet<typename std::remove_const<Packet>::type>::type& a) const
			{
				return p_to_f32(a);
			}
		};

		template<>
		struct functor_traits<scalar_cast_op<int32_t, float> >
		{
			enum { Cost = NumTraits<float>::AddCost, PacketAccess = 1 };
		};

		template<typename ArgType>
		struct unary_evaluator<CwiseUnaryOp<scalar_cast_op<int32_t, float>, ArgType>, IndexBased >
			: evaluator_base<CwiseUnaryOp<scalar_cast_op<int32_t, float>, ArgType> >
		{
			typedef CwiseUnaryOp<scalar_cast_op<int32_t, float>, ArgType> XprType;

			enum {
				CoeffReadCost = evaluator<ArgType>::CoeffReadCost + functor_traits<scalar_cast_op<int32_t, float>>::Cost,

				Flags = evaluator<ArgType>::Flags
				& (HereditaryBits | LinearAccessBit | (functor_traits<scalar_cast_op<int32_t, float>>::PacketAccess ? PacketAccessBit : 0)),
				Alignment = evaluator<ArgType>::Alignment
			};

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
				explicit unary_evaluator(const XprType& op)
				: m_functor(op.functor()),
				m_argImpl(op.nestedExpression())
			{
				EIGEN_INTERNAL_CHECK_COST_VALUE(NumTraits<float>::AddCost);
				EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
			}

			typedef typename XprType::CoeffReturnType CoeffReturnType;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
				CoeffReturnType coeff(Index row, Index col) const
			{
				return m_functor(m_argImpl.coeff(row, col));
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
				CoeffReturnType coeff(Index index) const
			{
				return m_functor(m_argImpl.coeff(index));
			}

			template<int LoadMode, typename PacketType>
			EIGEN_STRONG_INLINE
				PacketType packet(Index row, Index col) const
			{
				return m_functor.packetOp<PacketType>(m_argImpl.template packet<LoadMode, typename to_int_packet<PacketType>::type>(row, col));
			}

			template<int LoadMode, typename PacketType>
			EIGEN_STRONG_INLINE
				PacketType packet(Index index) const
			{
				return m_functor.packetOp<PacketType>(m_argImpl.template packet<LoadMode, typename to_int_packet<PacketType>::type>(index));
			}

		protected:
			const scalar_cast_op<int32_t, float> m_functor;
			evaluator<ArgType> m_argImpl;
		};

		template<typename Scalar, typename Rng> struct scalar_norm_dist_op {
			Rng rng;

			scalar_norm_dist_op(const Rng& _rng) : rng{ _rng } 
			{
			}

			scalar_norm_dist_op(const scalar_norm_dist_op& o)
				: rng{ o.rng }
			{
			}

			scalar_norm_dist_op(scalar_norm_dist_op&& o)
				: rng{ std::move(o.rng) }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const 
			{
				thread_local Scalar cache;
				thread_local bool valid = false;
				if (valid)
				{
					valid = false;
					return cache;
				}

				Scalar v1, v2, sx;
				while(1)
				{
					v1 = 2 * bit_to_ur(rng()) - 1;
					v2 = 2 * bit_to_ur(rng()) - 1;
					sx = v1 * v1 + v2 * v2;
					if (sx && sx < 1) break;
				}
				Scalar fx = std::sqrt((Scalar)-2.0 * std::log(sx) / sx);
				cache = fx * v2;
				valid = true;
				return fx * v1;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				thread_local Packet cache;
				thread_local bool valid = false;
				if (valid)
				{
					valid = false;
					return cache;
				}
				valid = true;
				return box_muller<Packet, Rng>{}(rng, cache);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_norm_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

	}

	template <typename Derived, typename T> EIGEN_DEVICE_FUNC inline 
		const CwiseBinaryOp<internal::scalar_lgamma_subt_op< typename internal::traits<Derived>::Scalar, T >, const Derived,
		const typename internal::plain_constant_type<Derived, T>::type>
		lgamma_subt(const Eigen::ArrayBase<Derived>& x, const T& scalar)  {
		
		return CwiseBinaryOp<internal::scalar_lgamma_subt_op< typename internal::traits<Derived>::Scalar, T >, const Derived,
			const typename internal::plain_constant_type<Derived, T>::type>(x.derived(), 
				typename internal::plain_constant_type<Derived, T>::type(x.derived().rows(), x.derived().cols(), internal::scalar_constant_op<T>(scalar))
			);
	}

	template<typename Derived, typename Derived2>
	inline const CwiseBinaryOp<internal::scalar_lgamma_subt_op<typename Derived::Scalar, typename Derived2::Scalar>, const Derived, const Derived2>
		lgamma_subt(const Eigen::ArrayBase<Derived>& x, const Eigen::ArrayBase<Derived2>& y)
	{
		return CwiseBinaryOp<internal::scalar_lgamma_subt_op<typename Derived::Scalar, typename Derived2::Scalar>, const Derived, const Derived2>(
			x.derived(),
			y.derived()
		);
	}

	template<typename Derived, typename Urng>
	inline const CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>
		norm_dist(Index rows, Index cols, Urng&& urng)
	{
		return CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
			rows, cols, internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
		);
	}
}
