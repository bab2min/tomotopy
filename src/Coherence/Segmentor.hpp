#pragma once

#include "Common.h"

namespace tomoto
{
	namespace coherence
	{
		template<Segmentation _seg, typename _CMFunc>
		class Segmentor;

		template<Segmentation _seg, typename _CMFunc>
		Segmentor<_seg, typename std::remove_reference<_CMFunc>::type>
			makeSegmentor(_CMFunc&& cm, const IProbEstimator* pe)
		{
			return { std::forward<_CMFunc>(cm), pe };
		}

		template<typename _CMFunc>
		class Segmentor<Segmentation::one_one, _CMFunc>
		{
			const IProbEstimator* pe;
			_CMFunc cm;
		public:
			Segmentor(const _CMFunc& _cm, const IProbEstimator* _pe) : cm{ _cm }, pe{ _pe }
			{
			}

			template<typename _TargetIter>
			double operator()(_TargetIter wordFirst, _TargetIter wordLast)
			{
				double ret = 0;
				double n = 0;
				for (auto it1 = wordFirst; it1 != wordLast; ++it1)
				{
					for (auto it2 = wordFirst; it2 != wordLast; ++it2)
					{
						if (it1 == it2) continue;
						ret += cm(pe, *it1, *it2);
						n += 1;
					}
				}
				return ret / n;
			}
		};

		template<typename _CMFunc>
		class Segmentor<Segmentation::one_pre, _CMFunc>
		{
			const IProbEstimator* pe;
			_CMFunc cm;
		public:
			Segmentor(const _CMFunc& _cm, const IProbEstimator* _pe) : cm{ _cm }, pe{ _pe }
			{
			}

			template<typename _TargetIter>
			double operator()(_TargetIter wordFirst, _TargetIter wordLast)
			{
				double ret = 0;
				double n = 0;
				for (auto it1 = wordFirst; it1 != wordLast; ++it1)
				{
					for (auto it2 = wordFirst; it2 != it1; ++it2)
					{
						ret += cm(pe, *it1, *it2);
						n += 1;
					}
				}
				return ret / n;
			}
		};

		template<typename _CMFunc>
		class Segmentor<Segmentation::one_suc, _CMFunc>
		{
			const IProbEstimator* pe;
			_CMFunc cm;
		public:
			Segmentor(const _CMFunc& _cm, const IProbEstimator* _pe) : cm{ _cm }, pe{ _pe }
			{
			}

			template<typename _TargetIter>
			double operator()(_TargetIter wordFirst, _TargetIter wordLast)
			{
				double ret = 0;
				double n = 0;
				for (auto it1 = wordFirst; it1 != wordLast; ++it1)
				{
					for (auto it2 = it1 + 1; it2 == wordLast; ++it2)
					{
						ret += cm(pe, *it1, *it2);
						n += 1;
					}
				}
				return ret / n;
			}
		};

		template<typename _CMFunc>
		class Segmentor<Segmentation::one_set, _CMFunc>
		{
			const IProbEstimator* pe;
			_CMFunc cm;
		public:
			Segmentor(const _CMFunc& _cm, const IProbEstimator* _pe) : cm{ _cm }, pe{ _pe }
			{
			}

			template<typename _TargetIter>
			double operator()(_TargetIter wordFirst, _TargetIter wordLast)
			{
				double ret = 0;
				double n = 0;
				for (auto it1 = wordFirst; it1 != wordLast; ++it1)
				{
					ret += cm(pe, *it1, std::vector<Vid>{ wordFirst, wordLast });
					n += 1;
				}
				return ret / n;
			}
		};


		template<typename _CMFunc>
		class Segmentor<Segmentation::one_all, _CMFunc>
		{
			const IProbEstimator* pe;
			_CMFunc cm;
		public:
			Segmentor(const _CMFunc& _cm, const IProbEstimator* _pe) : cm{ _cm }, pe{ _pe }
			{
			}

			template<typename _TargetIter>
			double operator()(_TargetIter wordFirst, _TargetIter wordLast)
			{
				double ret = 0;
				double n = 0;
				for (auto it1 = wordFirst; it1 != wordLast; ++it1)
				{
					std::vector<Vid> rest;
					rest.insert(rest.end(), wordFirst, it1);
					rest.insert(rest.end(), it1 + 1, wordLast);
					ret += cm(pe, *it1, rest);
					n += 1;
				}
				return ret / n;
			}
		};
	}
}
