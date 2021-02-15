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
			size_t burnIn
		)
		{
			const size_t K = ret.size();
			Eigen::Matrix<_Ty, -1, -1> l = multiNormal.getCovL();
			ret = (lowerBound + upperBound) / 2;
			Eigen::Matrix<_Ty, -1, 1> z = l.template triangularView<Eigen::Lower>().solve(ret - multiNormal.mean),
				a = lowerBound - multiNormal.mean,
				b = upperBound - multiNormal.mean,
				t, at, bt;
			for (size_t i = 0; i < burnIn; ++i)
			{
				for (size_t j = 0; j < K; ++j)
				{
					auto lj = l.col(j);
					z[j] = 0;
					t = l * z;
					_Ty lower_pos = -INFINITY, upper_pos = INFINITY,
						lower_neg = -INFINITY, upper_neg = INFINITY;
					at = ((a - t).array() / lj.array()).matrix();
					bt = ((b - t).array() / lj.array()).matrix();
					for (size_t k = 0; k < K; ++k)
					{
						if (lj[k] > 0)
						{
							lower_pos = std::max(lower_pos, at[k]);
							upper_pos = std::min(upper_pos, bt[k]);
						}
						else if (lj[k] < 0)
						{
							lower_neg = std::max(lower_neg, bt[k]);
							upper_neg = std::min(upper_neg, at[k]);
						}
					}
					z[j] = rtnorm::rtnorm(rng, std::max(lower_pos, lower_neg), std::min(upper_pos, upper_neg));
				}
			}
			ret = (l * z) + multiNormal.mean;
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