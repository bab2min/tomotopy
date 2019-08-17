#pragma once
#include <numeric>
#include "MultiNormalDistribution.hpp"
#include "rtnorm.hpp"

namespace tomoto
{
	namespace math
	{
		template<typename _Ty, typename _Out, typename _Rng>
		_Out sampleFromTruncatedMultiNormal(
			_Out ret,
			const MultiNormalDistribution<_Ty>& multiNormal,
			const Eigen::Matrix<_Ty, -1, 1>& lowerBound,
			const Eigen::Matrix<_Ty, -1, 1>& upperBound,
			_Rng& rng,
			size_t iteration)
		{
			constexpr _Ty epsilon = 1e-6;
			const size_t K = ret.size();
			Eigen::Matrix<_Ty, -1, 1> bias = Eigen::Matrix<_Ty, -1, 1>::Zero(K), lowers, uppers;
			auto& l = multiNormal.getCovL();
			ret.setZero();

			std::vector<size_t> ks(K);
			std::iota(ks.begin(), ks.end(), 0);
			for (size_t i = 0; i < iteration; ++i)
			{
				// shuffle sampling orders except during initialization
				if (i) std::shuffle(ks.begin(), ks.end(), rng);
				for (size_t kx = 0; kx < K; ++kx)
				{
					size_t k = ks[kx];
					ret[k] = 0;
					//bias = multiNormal.mean + l * ret;
					//bias.tail(K - k) = multiNormal.mean.tail(K - k) + l.block(k, 0, K - k, K) * ret;
					bias.tail(K - k) = multiNormal.mean.tail(K - k);
					bias.tail(K - k).noalias() += l.block(k, 0, K - k, K) * ret;
					lowers = (lowerBound - bias).tail(K - k).array() / l.col(k).tail(K - k).array();
					uppers = (upperBound - bias).tail(K - k).array() / l.col(k).tail(K - k).array();
					_Ty nLower = lowers[0], nUpper = uppers[0];
					if (l(k, k) < 0) std::swap(nLower, nUpper);
					if (i)
					{
						for (size_t j = 1; j < lowers.size(); ++j)
						{
							if (l.col(k)(j + k) > epsilon)
							{
								if (lowers[j] > nLower) nLower = lowers[j];
								if (uppers[j] < nUpper) nUpper = uppers[j];
							}
							else if (l.col(k)(j + k) < -epsilon)
							{
								if (uppers[j] > nLower) nLower = uppers[j];
								if (lowers[j] < nUpper) nUpper = lowers[j];
							}
						}
					}
					if (abs(nLower - nUpper) <= 1e-4) ret[k] = (nLower + nUpper) / 2;
					else
					{
						ret[k] = rtnorm::rtnorm(rng, nLower, nUpper);
					}
				}
			}
			ret = l * ret;
			ret += multiNormal.mean;
			return ret;
		}

		template<typename _Ty, typename _Out, typename _Rng>
		_Out sampleFromTruncatedMultiNormalRejection(
			_Out ret,
			const MultiNormalDistribution<_Ty>& multiNormal,
			const Eigen::Matrix<_Ty, -1, 1>& lowerBound,
			const Eigen::Matrix<_Ty, -1, 1>& upperBound,
			_Rng& rng)
		{
			const size_t K = ret.size();
			auto& l = multiNormal.getCovL();
			std::normal_distribution<_Ty> normal{};
			while (1)
			{
				for (size_t k = 0; k < K; ++k) ret[k] = normal(rng);
				ret = l * ret;
				ret += multiNormal.mean;
				if ((lowerBound.array() <= ret.array()).all() && (ret.array() <= upperBound.array()).all())
				{
					return ret;
				}
			}
		}
	}
}