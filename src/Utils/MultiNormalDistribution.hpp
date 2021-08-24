#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "serializer.hpp"

namespace tomoto
{
	namespace math
	{
		template<typename _Ty = float>
		struct MultiNormalDistribution
		{
			static constexpr _Ty log2pi = (_Ty)1.83787706641;
			Eigen::Matrix<_Ty, -1, 1> mean;
			Eigen::Matrix<_Ty, -1, -1> cov, l;
			_Ty logDet = 0;

			MultiNormalDistribution(size_t k = 0) :
				mean{ Eigen::Matrix<_Ty, -1, 1>::Zero(k) },
				cov{ Eigen::Matrix<_Ty, -1, -1>::Identity(k, k) },
				l{ Eigen::Matrix<_Ty, -1, -1>::Identity(k, k) }
			{
			}

			_Ty getLL(const Eigen::Matrix<_Ty, -1, 1>& x) const
			{
				_Ty ll = -((x - mean).transpose() * cov.inverse() * (x - mean))[0] / 2;
				ll -= log2pi * mean.size() / 2 + logDet;
				return ll;
			}

			const Eigen::Matrix<_Ty, -1, -1>& getCovL() const
			{
				return l;
			}

			template<typename _List>
			static MultiNormalDistribution<_Ty> estimate(_List list, size_t len)
			{
				MultiNormalDistribution<_Ty> newDist;
				if (len)
				{
					newDist.mean = list(0);
					for (size_t i = 1; i < len; ++i) newDist.mean += list(i);
					newDist.mean /= len;
					newDist.cov = Eigen::Matrix<_Ty, -1, -1>::Identity(newDist.mean.size(), newDist.mean.size());
					for (size_t i = 0; i < len; ++i)
					{
						Eigen::Matrix<_Ty, -1, 1> o = list(i) - newDist.mean;
						newDist.cov += o * o.transpose();
					}
					if (len > 1) newDist.cov /= len - 1;
				}
				Eigen::MatrixXd l = newDist.cov.template cast<double>().llt().matrixL();
				newDist.l = l.template cast<float>();
				newDist.logDet = l.diagonal().array().log().sum();
				return newDist;
			}

			DEFINE_SERIALIZER_CALLBACK(onRead, mean, cov);
		private:
			void onRead() 
			{
				l = cov.llt().matrixL();
				logDet = l.diagonal().array().log().sum();
			}
		};

	}
}