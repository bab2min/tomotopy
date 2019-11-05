#pragma once
#include <type_traits>
#include <Eigen/Dense>
#include "math.h"

#ifdef EIGEN_VECTORIZE_AVX
#include <immintrin.h>
#include "avx_gamma.h"

namespace Eigen
{
	namespace internal
	{
		template<typename PacketType>
		struct to_int_packet
		{
			typedef PacketType type;
		};

		template<> struct to_int_packet<Packet8f>
		{
			typedef Packet8i type;
		};

		template<> struct to_int_packet<Packet4f>
		{
			typedef Packet4i type;
		};

		EIGEN_STRONG_INLINE Packet8f p_to_f32(const Packet8i& a)
		{
			return _mm256_cvtepi32_ps(a);
		}

		EIGEN_STRONG_INLINE Packet4f p_to_f32(const Packet4i& a)
		{
			return _mm_cvtepi32_ps(a);
		}
	}
}

#elif defined(EIGEN_VECTORIZE_SSE2)
#include <xmmintrin.h>
#include "sse_gamma.h"

namespace Eigen
{
	namespace internal
	{
		template<typename PacketType>
		struct to_int_packet
		{
			typedef PacketType type;
		};

		template<> struct to_int_packet<Packet4f>
		{
			typedef Packet4i type;
		};

		EIGEN_STRONG_INLINE Packet4f p_to_f32(const Packet4i& a)
		{
			return _mm_cvtepi32_ps(a);
		}
	}
}
#else
template<typename PacketType>
struct to_int_packet
{
	typedef PacketType type;
};

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
}
