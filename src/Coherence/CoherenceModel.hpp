#pragma once

/*
* Röder, M., Both, A., & Hinneburg, A. (2015, February). Exploring the space of topic coherence measures. In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399-408).
http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
https://github.com/dice-group/Palmetto

*/

#include "Common.h"
#include "ConfirmMeasurer.hpp"
#include "ProbEstimator.hpp"
#include "Segmentor.hpp"

namespace tomoto
{
	namespace coherence
	{
		class CoherenceModel
		{
			std::unique_ptr<IProbEstimator> pe;
			ProbEstimation pe_type = ProbEstimation::none;

			template<ProbEstimation _pe>
			void init(size_t windowSize)
			{
				pe_type = _pe;
				pe = std::make_unique<ProbEstimator<_pe>>(windowSize);
			}

			template<ProbEstimation _pe, typename _TargetIter>
			void _insertTargets(_TargetIter targetFirst, _TargetIter targetLast)
			{
				((ProbEstimator<_pe>*)pe.get())->insertTargets(targetFirst, targetLast);
			}

			template<ProbEstimation _pe, typename _TargetIter>
			void _insertDoc(_TargetIter wordFirst, _TargetIter wordLast)
			{
				((ProbEstimator<_pe>*)pe.get())->insertDoc(wordFirst, wordLast);
			}

		public:
			CoherenceModel() = default;

			CoherenceModel(ProbEstimation _pe, size_t windowSize)
			{
				switch (_pe)
				{
				case ProbEstimation::document:
					init<ProbEstimation::document>(windowSize);
					break;
				case ProbEstimation::sliding_windows:
					init<ProbEstimation::sliding_windows>(windowSize);
					break;
				default:
					throw std::invalid_argument{ "invalid ProbEstimation `_pe`" };
				}
			}

			template<typename _TargetIter>
			void insertTargets(_TargetIter targetFirst, _TargetIter targetLast)
			{
				switch (pe_type)
				{
				case ProbEstimation::document:
					return _insertTargets<ProbEstimation::document>(targetFirst, targetLast);
				case ProbEstimation::sliding_windows:
					return _insertTargets<ProbEstimation::sliding_windows>(targetFirst, targetLast);
				default:
					throw std::invalid_argument{ "invalid ProbEstimation `_pe`" };
				}
			}

			template<typename _TargetIter>
			void insertDoc(_TargetIter wordFirst, _TargetIter wordLast)
			{
				switch (pe_type)
				{
				case ProbEstimation::document:
					return _insertDoc<ProbEstimation::document>(wordFirst, wordLast);
				case ProbEstimation::sliding_windows:
					return _insertDoc<ProbEstimation::sliding_windows>(wordFirst, wordLast);
				default:
					throw std::invalid_argument{ "invalid ProbEstimation `_pe`" };
				}
			}

			template<Segmentation _seg, typename _CMFunc, typename _TargetIter>
			double getScore(_CMFunc&& cm, _TargetIter targetFirst, _TargetIter targetLast) const
			{
				return makeSegmentor<_seg>(std::forward<_CMFunc>(cm), pe.get())(targetFirst, targetLast);
			}

		};
	}
}
