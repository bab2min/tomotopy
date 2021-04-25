#pragma once

#include <set>
#include "Labeler.h"
#include "../Utils/EigenAddonOps.hpp"
#include "../Utils/Trie.hpp"
#include "../Utils/ThreadPool.hpp"

/*
Implementation of First-order Relevance for topic labeling by bab2min

* Mei, Q., Shen, X., & Zhai, C. (2007, August). Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499).

*/

namespace tomoto
{
	namespace label
	{
		class PMIExtractor : public IExtractor
		{
			size_t candMinCnt, candMinDf, minLabelLen, maxLabelLen, maxCandidates;
			bool normalized;
		public:
			PMIExtractor(size_t _candMinCnt = 10, size_t _candMinDf = 2, 
				size_t _minLabelLen = 1, size_t _maxLabelLen = 5, size_t _maxCandidates = 1000,
				bool _normalized = false
			)
				: candMinCnt{ _candMinCnt }, candMinDf{ _candMinDf }, 
				minLabelLen{ _minLabelLen }, maxLabelLen{ _maxLabelLen }, 
				maxCandidates{ _maxCandidates }, normalized{ _normalized }
			{
			}

			std::vector<Candidate> extract(const ITopicModel* tm) const override;
		};

		class PMIBEExtractor : public IExtractor
		{
			size_t candMinCnt, candMinDf, minLabelLen, maxLabelLen, maxCandidates;
		public:
			PMIBEExtractor(size_t _candMinCnt = 10, size_t _candMinDf = 2, 
				size_t _minLabelLen = 1, size_t _maxLabelLen = 5, size_t _maxCandidates = 1000
			)
				: candMinCnt{ _candMinCnt }, candMinDf{ _candMinDf }, minLabelLen{ _minLabelLen }, maxLabelLen{ _maxLabelLen }, maxCandidates{ _maxCandidates }
			{
			}

			std::vector<Candidate> extract(const ITopicModel* tm) const override;
		};

		class FoRelevance : public ILabeler
		{
			struct CandidateEx : public Candidate
			{
				std::unordered_map<std::string, size_t> names;
				std::set<size_t> docIds;
				Eigen::Array<Float, -1, 1> scores;

				CandidateEx()
				{
				}

				CandidateEx(const Candidate& c)
					: Candidate{ c }
				{
				}
			};

			const ITopicModel* tm;
			size_t candMinDf;
			float smoothing, lambda, mu;
			size_t windowSize;
			std::unique_ptr<ThreadPool> pool;
			std::unique_ptr<std::mutex[]> mtx;
			std::vector<CandidateEx> candidates;

			template<bool _lock>
			const Eigen::ArrayXi& updateContext(size_t docId, const tomoto::DocumentBase* doc, const Trie<Vid, size_t>* root);

			void estimateContexts();

		public:
			template<typename _Iter>
			FoRelevance(const ITopicModel* _tm,
				_Iter candFirst, _Iter candEnd,
				size_t _candMinDf = 2, float _smoothing = 0.1f, float _lambda = 0.1f, float _mu = 0.1f,
				size_t _windowSize = (size_t)-1,
				size_t numWorkers = 0)
				: tm{ _tm }, candMinDf{ _candMinDf },
				smoothing{ _smoothing }, lambda{ _lambda }, mu{ _mu }, windowSize{ _windowSize }
			{
				if (!numWorkers) numWorkers = std::thread::hardware_concurrency();
				if (numWorkers > 1)
				{
					pool = std::make_unique<ThreadPool>(numWorkers);
					mtx = std::make_unique<std::mutex[]>(numWorkers);
				}

				for (; candFirst != candEnd; ++candFirst)
				{
					candidates.emplace_back(*candFirst);
				}

				estimateContexts();
			}

			std::vector<std::pair<std::string, float>> getLabels(Tid tid, size_t topK = 10) const override;
		};
	}
}
