#pragma once

#include <set>
#include "Labeler.h"
#include "../Utils/EigenAddonOps.hpp"
#include "../Utils/Trie.hpp"

/*
Implementation of First-order Relevance for topic labeling by bab2min

* Mei, Q., Shen, X., & Zhai, C. (2007, August). Automatic labeling of multinomial topic models. In Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 490-499).

*/

namespace tomoto
{
	namespace label
	{
		template<typename _DocIter, typename _Freqs>
		std::vector<Candidate> extractPMINgrams(_DocIter docBegin, _DocIter docEnd, 
			_Freqs&& vocabFreqs, _Freqs&& vocabDf,
			size_t candMinCnt, size_t candMinDf, size_t minNgrams, size_t maxNgrams, size_t maxCandidates, float minScore)
		{
			struct vvhash
			{
				size_t operator()(const std::pair<Vid, Vid>& k) const
				{
					return std::hash<Vid>{}(k.first) ^ std::hash<Vid>{}(k.second);
				}
			};

			// counting unigrams & bigrams
			std::unordered_map<std::pair<Vid, Vid>, size_t, vvhash> bigramCnt, bigramDf;

			for(auto docIt = docBegin; docIt != docEnd; ++docIt)
			{
				std::unordered_set<std::pair<Vid, Vid>, vvhash> uniqBigram;
				auto doc = *docIt;
				Vid prevWord = doc[0];
				for (size_t j = 1; j < doc.size(); ++j)
				{
					Vid curWord = doc[j];
					if (curWord != non_vocab_id && vocabFreqs[curWord] >= candMinCnt && vocabDf[curWord] >= candMinDf)
					{
						if (prevWord != non_vocab_id && vocabFreqs[prevWord] >= candMinCnt && vocabDf[prevWord] >= candMinDf)
						{
							bigramCnt[std::make_pair(prevWord, curWord)]++;
							uniqBigram.emplace(prevWord, curWord);
						}
					}
					prevWord = curWord;
				}

				for (auto& p : uniqBigram) bigramDf[p]++;
			}


			// counting ngrams
			std::vector<TrieEx<Vid, size_t>> trieNodes;

			if (maxNgrams > 2)
			{
				std::unordered_set<std::pair<Vid, Vid>, vvhash> validPair;
				for (auto& p : bigramCnt)
				{
					if (p.second >= candMinCnt) validPair.emplace(p.first);
				}

				trieNodes.resize(1);
				auto allocNode = [&]() { return trieNodes.emplace_back(), & trieNodes.back(); };

				for (auto docIt = docBegin; docIt != docEnd; ++docIt)
				{
					auto doc = *docIt;
					if (trieNodes.capacity() < trieNodes.size() + doc.size() * maxNgrams)
					{
						trieNodes.reserve(std::max(trieNodes.size() + doc.size() * maxNgrams, trieNodes.capacity() * 2));
					}

					Vid prevWord = doc[0];
					size_t labelLen = 0;
					auto node = &trieNodes[0];
					if (prevWord != non_vocab_id && vocabFreqs[prevWord] >= candMinCnt)
					{
						node = trieNodes[0].makeNext(prevWord, allocNode);
						node->val++;
						labelLen = 1;
					}

					for (size_t j = 1; j < doc.size(); ++j)
					{
						Vid curWord = doc[j];

						if (curWord != non_vocab_id && vocabFreqs[curWord] < candMinCnt)
						{
							node = &trieNodes[0];
							labelLen = 0;
						}
						else
						{
							if (labelLen >= maxNgrams)
							{
								node = node->getFail();
								labelLen--;
							}

							if (validPair.count(std::make_pair(prevWord, curWord)))
							{
								auto nnode = node->makeNext(curWord, allocNode);
								node = nnode;
								do
								{
									nnode->val++;
								} while (nnode = nnode->getFail());
								labelLen++;
							}
							else
							{
								node = trieNodes[0].makeNext(curWord, allocNode);
								node->val++;
								labelLen = 1;
							}
						}
						prevWord = curWord;
					}
				}
			}

			float totN = std::accumulate(vocabFreqs.begin(), vocabFreqs.end(), (size_t)0);

			// calculating PMIs
			std::vector<Candidate> candidates;
			for (auto& p : bigramCnt)
			{
				auto& bigram = p.first;
				if (p.second < candMinCnt) continue;
				if (bigramDf[bigram] < candMinDf) continue;
				auto pmi = std::log(p.second * totN
					/ vocabFreqs[bigram.first] / vocabFreqs[bigram.second]);
				if (pmi <= 0) continue;
				candidates.emplace_back(pmi, bigram.first, bigram.second);
			}

			if (maxNgrams > 2)
			{
				std::vector<Vid> rkeys;
				trieNodes[0].traverse_with_keys([&](const TrieEx<Vid, size_t>* node, const std::vector<Vid>& rkeys)
				{
					if (rkeys.size() <= 2 || rkeys.size() < minNgrams || node->val < candMinCnt) return;
					auto pmi = node->val / totN;
					for (auto k : rkeys)
					{
						pmi *= totN / vocabFreqs[k];
					}
					pmi = std::log(pmi);
					if (pmi < minScore) return;
					candidates.emplace_back(pmi, rkeys);
				}, rkeys);
			}

			std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b)
			{
				return a.score > b.score;
			});
			if (candidates.size() > maxCandidates) candidates.erase(candidates.begin() + maxCandidates, candidates.end());
			return candidates;
		}


		class PMIExtractor : public IExtractor
		{
			size_t candMinCnt, candMinDf, minLabelLen, maxLabelLen, maxCandidates;
		public:
			PMIExtractor(size_t _candMinCnt = 10, size_t _candMinDf = 2, size_t _minLabelLen = 1, size_t _maxLabelLen = 5, size_t _maxCandidates = 1000)
				: candMinCnt{ _candMinCnt }, candMinDf{ _candMinDf }, minLabelLen{ _minLabelLen}, maxLabelLen{ _maxLabelLen }, maxCandidates{ _maxCandidates }
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
					pool = make_unique<ThreadPool>(numWorkers);
					mtx = make_unique<std::mutex[]>(numWorkers);
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
