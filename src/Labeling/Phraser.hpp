#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include "Labeler.h"
#include "../Utils/Trie.hpp"

#ifdef TMT_USE_BTREE
#include "btree/map.h"
#else
#endif

namespace tomoto
{
	namespace phraser
	{
#ifdef TMT_USE_BTREE
		template<typename K, typename V> using map = btree::map<K, V>;
#else
		template<typename K, typename V> using map = std::map<K, V>;
#endif

		namespace detail
		{
			struct vvhash
			{
				size_t operator()(const std::pair<Vid, Vid>& k) const
				{
					return std::hash<Vid>{}(k.first) ^ std::hash<Vid>{}(k.second);
				}
			};
		}

		template<typename _DocIter>
		void countUnigrams(std::vector<size_t>& unigramCf, std::vector<size_t>& unigramDf,
			_DocIter docBegin, _DocIter docEnd
		)
		{
			for (auto docIt = docBegin; docIt != docEnd; ++docIt)
			{
				auto doc = *docIt;
				if (!doc.size()) continue;
				std::unordered_set<Vid> uniqs;
				for (size_t i = 0; i < doc.size(); ++i)
				{
					if (doc[i] == non_vocab_id) continue;
					unigramCf[doc[i]]++;
					uniqs.emplace(doc[i]);
				}

				for (auto w : uniqs) unigramDf[w]++;
			}
		}

		template<typename _DocIter, typename _Freqs>
		void countBigrams(map<std::pair<Vid, Vid>, size_t>& bigramCf,
			map<std::pair<Vid, Vid>, size_t>& bigramDf,
			_DocIter docBegin, _DocIter docEnd,
			_Freqs&& vocabFreqs, _Freqs&& vocabDf,
			size_t candMinCnt, size_t candMinDf
		)
		{
			for (auto docIt = docBegin; docIt != docEnd; ++docIt)
			{
				std::unordered_set<std::pair<Vid, Vid>, detail::vvhash> uniqBigram;
				auto doc = *docIt;
				if (!doc.size()) continue;
				Vid prevWord = doc[0];
				for (size_t j = 1; j < doc.size(); ++j)
				{
					Vid curWord = doc[j];
					if (curWord != non_vocab_id && vocabFreqs[curWord] >= candMinCnt && vocabDf[curWord] >= candMinDf)
					{
						if (prevWord != non_vocab_id && vocabFreqs[prevWord] >= candMinCnt && vocabDf[prevWord] >= candMinDf)
						{
							bigramCf[std::make_pair(prevWord, curWord)]++;
							uniqBigram.emplace(prevWord, curWord);
						}
					}
					prevWord = curWord;
				}

				for (auto& p : uniqBigram) bigramDf[p]++;
			}
		}

		template<bool _reverse, typename _DocIter, typename _Freqs, typename _BigramPairs>
		void countNgrams(std::vector<TrieEx<Vid, size_t>>& dest,
			_DocIter docBegin, _DocIter docEnd,
			_Freqs&& vocabFreqs, _Freqs&& vocabDf, _BigramPairs&& validPairs,
			size_t candMinCnt, size_t candMinDf, size_t maxNgrams
		)
		{
			if (dest.empty())
			{
				dest.resize(1);
				dest.reserve(1024);
			}
			auto allocNode = [&]() { return dest.emplace_back(), & dest.back(); };

			for (auto docIt = docBegin; docIt != docEnd; ++docIt)
			{
				auto doc = *docIt;
				if (!doc.size()) continue;
				if (dest.capacity() < dest.size() + doc.size() * maxNgrams)
				{
					dest.reserve(std::max(dest.size() + doc.size() * maxNgrams, dest.capacity() * 2));
				}

				Vid prevWord = _reverse ? *doc.rbegin() : *doc.begin();
				size_t labelLen = 0;
				auto node = &dest[0];
				if (prevWord != non_vocab_id && vocabFreqs[prevWord] >= candMinCnt && vocabDf[prevWord] >= candMinDf)
				{
					node = dest[0].makeNext(prevWord, allocNode);
					node->val++;
					labelLen = 1;
				}

				const auto func = [&](Vid curWord)
				{
					if (curWord != non_vocab_id && (vocabFreqs[curWord] < candMinCnt || vocabDf[curWord] < candMinDf))
					{
						node = &dest[0];
						labelLen = 0;
					}
					else
					{
						if (labelLen >= maxNgrams)
						{
							node = node->getFail();
							labelLen--;
						}

						if (validPairs.count(_reverse ? std::make_pair(curWord, prevWord) : std::make_pair(prevWord, curWord)))
						{
							auto nnode = node->makeNext(curWord, allocNode);
							node = nnode;
							do
							{
								nnode->val++;
							} while ((nnode = nnode->getFail()));
							labelLen++;
						}
						else
						{
							node = dest[0].makeNext(curWord, allocNode);
							node->val++;
							labelLen = 1;
						}
					}
					prevWord = curWord;
				};

				if (_reverse) std::for_each(doc.rbegin() + 1, doc.rend(), func);
				else std::for_each(doc.begin() + 1, doc.end(), func);
			}
		}

		inline void mergeNgramCounts(std::vector<TrieEx<Vid, size_t>>& dest, std::vector<TrieEx<Vid, size_t>>&& src)
		{
			if (src.empty()) return;
			if (dest.empty()) dest.resize(1);

			auto allocNode = [&]() { return dest.emplace_back(), & dest.back(); };

			std::vector<Vid> rkeys;
			src[0].traverse_with_keys([&](const TrieEx<Vid, size_t>* node, const std::vector<Vid>& rkeys)
			{
				if (dest.capacity() < dest.size() + rkeys.size() * rkeys.size())
				{
					dest.reserve(std::max(dest.size() + rkeys.size() * rkeys.size(), dest.capacity() * 2));
				}
				dest[0].build(rkeys.begin(), rkeys.end(), 0, allocNode)->val += node->val;
			}, rkeys);
		}

		inline float branchingEntropy(const TrieEx<Vid, size_t>* node, size_t minCnt)
		{
			float entropy = 0;
			size_t rest = node->val;
			for (auto n : *node)
			{
				float p = n.second->val / (float)node->val;
				entropy -= p * std::log(p);
				rest -= n.second->val;
			}
			if (rest > 0)
			{
				float p = rest / (float)node->val;
				entropy -= p * std::log(std::min(std::max(minCnt, (size_t)1), (size_t)rest) / (float)node->val);
			}
			return entropy;
		}

		template<typename _LocalData, typename _ReduceFn>
		_LocalData parallelReduce(std::vector<_LocalData>&& data, _ReduceFn&& fn, ThreadPool* pool = nullptr)
		{
			if (pool && pool->getNumWorkers() > 1)
			{
				for (size_t s = data.size(); s > 1; s = (s + 1) / 2)
				{
					std::vector<std::future<void>> futures;
					size_t h = (s + 1) / 2;
					for (size_t i = h; i < s; ++i)
					{
						futures.emplace_back(pool->enqueue([&, i, h](size_t)
						{
							_LocalData d = std::move(data[i]);
							fn(data[i - h], std::move(d));
						}));
					}
					for (auto& f : futures) f.get();
				}
			}
			else
			{
				for (size_t i = 1; i < data.size(); ++i)
				{
					_LocalData d = std::move(data[i]);
					fn(data[0], std::move(d));
				}
			}
			return std::move(data[0]);
		}

		template<typename _DocIter, typename _Freqs>
		std::vector<label::Candidate> extractPMINgrams(_DocIter docBegin, _DocIter docEnd,
			_Freqs&& vocabFreqs, _Freqs&& vocabDf,
			size_t candMinCnt, size_t candMinDf, size_t minNgrams, size_t maxNgrams, size_t maxCandidates,
			float minScore, bool normalized = false,
			ThreadPool* pool = nullptr)
		{
			// counting unigrams & bigrams
			map<std::pair<Vid, Vid>, size_t> bigramCnt, bigramDf;

			if (pool && pool->getNumWorkers() > 1)
			{
				using LocalCfDf = std::pair<
					decltype(bigramCnt),
					decltype(bigramDf)
				>;
				std::vector<LocalCfDf> localdata(pool->getNumWorkers());
				std::vector<std::future<void>> futures;
				const size_t stride = pool->getNumWorkers() * 8;
				auto docIt = docBegin;
				for (size_t i = 0; i < stride && docIt != docEnd; ++i, ++docIt)
				{
					futures.emplace_back(pool->enqueue([&, docIt, stride](size_t tid)
					{
						countBigrams(localdata[tid].first, localdata[tid].second, makeStrideIter(docIt, stride, docEnd), makeStrideIter(docEnd, stride, docEnd), vocabFreqs, vocabDf, candMinCnt, candMinDf);
					}));
				}

				for (auto& f : futures) f.get();

				auto r = parallelReduce(std::move(localdata), [](LocalCfDf& dest, LocalCfDf&& src)
				{
					for (auto& p : src.first) dest.first[p.first] += p.second;
					for (auto& p : src.second) dest.second[p.first] += p.second;
				}, pool);

				bigramCnt = std::move(r.first);
				bigramDf = std::move(r.second);
			}
			else
			{
				countBigrams(bigramCnt, bigramDf, docBegin, docEnd, vocabFreqs, vocabDf, candMinCnt, candMinDf);
			}

			// counting ngrams
			std::vector<TrieEx<Vid, size_t>> trieNodes;
			if (maxNgrams > 2)
			{
				std::unordered_set<std::pair<Vid, Vid>, detail::vvhash> validPairs;
				for (auto& p : bigramCnt)
				{
					if (p.second >= candMinCnt && bigramDf[p.first] >= candMinDf) validPairs.emplace(p.first);
				}

				if (pool && pool->getNumWorkers() > 1)
				{
					using LocalFwBw = std::vector<TrieEx<Vid, size_t>>;
					std::vector<LocalFwBw> localdata(pool->getNumWorkers());
					std::vector<std::future<void>> futures;
					const size_t stride = pool->getNumWorkers() * 8;
					auto docIt = docBegin;
					for (size_t i = 0; i < stride && docIt != docEnd; ++i, ++docIt)
					{
						futures.emplace_back(pool->enqueue([&, docIt, stride](size_t tid)
						{
							countNgrams<false>(localdata[tid],
								makeStrideIter(docIt, stride, docEnd),
								makeStrideIter(docEnd, stride, docEnd),
								vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams
							);
						}));
					}

					for (auto& f : futures) f.get();

					auto r = parallelReduce(std::move(localdata), [&](LocalFwBw& dest, LocalFwBw&& src)
					{
						mergeNgramCounts(dest, std::move(src));
					}, pool);

					trieNodes = std::move(r);
				}
				else
				{
					countNgrams<false>(trieNodes,
						docBegin, docEnd,
						vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams
					);
				}
			}

			float totN = (float)std::accumulate(vocabFreqs.begin(), vocabFreqs.end(), (size_t)0);
			const float logTotN = std::log(totN);

			// calculating PMIs
			std::vector<label::Candidate> candidates;
			for (auto& p : bigramCnt)
			{
				auto& bigram = p.first;
				if (p.second < candMinCnt) continue;
				if (bigramDf[bigram] < candMinDf) continue;
				auto pmi = std::log(p.second * totN
					/ vocabFreqs[bigram.first] / vocabFreqs[bigram.second]);
				if (normalized)
				{
					pmi /= std::log(totN / p.second);
				}
				if (pmi < minScore) continue;
				candidates.emplace_back(pmi, bigram.first, bigram.second);
				candidates.back().cf = p.second;
				candidates.back().df = bigramDf[bigram];
			}

			if (maxNgrams > 2)
			{
				std::vector<Vid> rkeys;
				trieNodes[0].traverse_with_keys([&](const TrieEx<Vid, size_t>* node, const std::vector<Vid>& rkeys)
				{
					if (rkeys.size() <= 2 || rkeys.size() < minNgrams || rkeys.size() > maxNgrams || node->val < candMinCnt) return;
					auto pmi = std::log((float)node->val) - logTotN;
					for (auto k : rkeys)
					{
						pmi += logTotN - std::log((float)vocabFreqs[k]);
					}
					if (normalized)
					{
						pmi /= (logTotN - std::log((float)node->val)) * (rkeys.size() - 1);
					}
					if (pmi < minScore) return;
					candidates.emplace_back(pmi, rkeys);
					candidates.back().cf = node->val;
				}, rkeys);
			}

			std::sort(candidates.begin(), candidates.end(), [](const label::Candidate& a, const label::Candidate& b)
			{
				return a.score > b.score;
			});
			if (candidates.size() > maxCandidates) candidates.erase(candidates.begin() + maxCandidates, candidates.end());
			return candidates;
		}

		template<typename _DocIter, typename _Freqs>
		std::vector<label::Candidate> extractPMIBENgrams(_DocIter docBegin, _DocIter docEnd,
			_Freqs&& vocabFreqs, _Freqs&& vocabDf,
			size_t candMinCnt, size_t candMinDf, size_t minNgrams, size_t maxNgrams, size_t maxCandidates,
			float minNPMI = 0, float minNBE = 0,
			ThreadPool* pool = nullptr)
		{
			// counting unigrams & bigrams
			map<std::pair<Vid, Vid>, size_t> bigramCnt, bigramDf;

			if (pool && pool->getNumWorkers() > 1)
			{
				using LocalCfDf = std::pair<
					decltype(bigramCnt),
					decltype(bigramDf)
				>;
				std::vector<LocalCfDf> localdata(pool->getNumWorkers());
				std::vector<std::future<void>> futures;
				const size_t stride = pool->getNumWorkers() * 8;
				auto docIt = docBegin;
				for (size_t i = 0; i < stride && docIt != docEnd; ++i, ++docIt)
				{
					futures.emplace_back(pool->enqueue([&, docIt, stride](size_t tid)
					{
						countBigrams(localdata[tid].first, localdata[tid].second, makeStrideIter(docIt, stride, docEnd), makeStrideIter(docEnd, stride, docEnd), vocabFreqs, vocabDf, candMinCnt, candMinDf);
					}));
				}

				for (auto& f : futures) f.get();

				auto r = parallelReduce(std::move(localdata), [](LocalCfDf& dest, LocalCfDf&& src)
				{
					for (auto& p : src.first) dest.first[p.first] += p.second;
					for (auto& p : src.second) dest.second[p.first] += p.second;
				}, pool);

				bigramCnt = std::move(r.first);
				bigramDf = std::move(r.second);
			}
			else
			{
				countBigrams(bigramCnt, bigramDf, docBegin, docEnd, vocabFreqs, vocabDf, candMinCnt, candMinDf);
			}

			// counting ngrams
			std::vector<TrieEx<Vid, size_t>> trieNodes, trieNodesBw;
			if (maxNgrams > 2)
			{
				std::unordered_set<std::pair<Vid, Vid>, detail::vvhash> validPairs;
				for (auto& p : bigramCnt)
				{
					if (p.second >= candMinCnt && bigramDf[p.first] >= candMinDf) validPairs.emplace(p.first);
				}

				if (pool && pool->getNumWorkers() > 1)
				{
					using LocalFwBw = std::pair<
						std::vector<TrieEx<Vid, size_t>>,
						std::vector<TrieEx<Vid, size_t>>
					>;
					std::vector<LocalFwBw> localdata(pool->getNumWorkers());
					std::vector<std::future<void>> futures;
					const size_t stride = pool->getNumWorkers() * 8;
					auto docIt = docBegin;
					for (size_t i = 0; i < stride && docIt != docEnd; ++i, ++docIt)
					{
						futures.emplace_back(pool->enqueue([&, docIt, stride](size_t tid)
						{
							countNgrams<false>(localdata[tid].first, 
								makeStrideIter(docIt, stride, docEnd), 
								makeStrideIter(docEnd, stride, docEnd), 
								vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams + 1
							);
							countNgrams<true>(localdata[tid].second, 
								makeStrideIter(docIt, stride, docEnd), 
								makeStrideIter(docEnd, stride, docEnd), 
								vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams + 1
							);
						}));
					}

					for (auto& f : futures) f.get();

					auto r = parallelReduce(std::move(localdata), [&](LocalFwBw& dest, LocalFwBw&& src)
					{
						mergeNgramCounts(dest.first, std::move(src.first));
						mergeNgramCounts(dest.second, std::move(src.second));
					}, pool);

					trieNodes = std::move(r.first);
					trieNodesBw = std::move(r.second);
				}
				else
				{
					countNgrams<false>(trieNodes, 
						docBegin, docEnd, 
						vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams + 1
					);
					countNgrams<true>(trieNodesBw, 
						docBegin, docEnd, 
						vocabFreqs, vocabDf, validPairs, candMinCnt, candMinDf, maxNgrams + 1
					);
				}
			}

			float totN = std::accumulate(vocabFreqs.begin(), vocabFreqs.end(), (size_t)0);
			const float logTotN = std::log(totN);

			// calculating PMIs
			std::vector<label::Candidate> candidates;
			for (auto& p : bigramCnt)
			{
				auto& bigram = p.first;
				if (p.second < candMinCnt) continue;
				if (bigramDf[bigram] < candMinDf) continue;
				float npmi = std::log(p.second * totN
					/ vocabFreqs[bigram.first] / vocabFreqs[bigram.second]);
				npmi /= std::log(totN / p.second);
				if (npmi < minNPMI) continue;

				float rbe = branchingEntropy(trieNodes[0].getNext(bigram.first)->getNext(bigram.second), candMinCnt);
				float lbe = branchingEntropy(trieNodesBw[0].getNext(bigram.second)->getNext(bigram.first), candMinCnt);
				float nbe = std::sqrt(rbe * lbe) / (float)std::log(p.second);
				if (nbe < minNBE) continue;
				candidates.emplace_back(npmi * nbe, bigram.first, bigram.second);
				candidates.back().cf = p.second;
				candidates.back().df = bigramDf[bigram];
			}

			if (maxNgrams > 2)
			{
				std::vector<Vid> rkeys;
				trieNodes[0].traverse_with_keys([&](const TrieEx<Vid, size_t>* node, const std::vector<Vid>& rkeys)
				{
					if (rkeys.size() <= 2 || rkeys.size() < minNgrams || rkeys.size() > maxNgrams || node->val < candMinCnt) return;
					auto npmi = std::log((float)node->val) - logTotN;
					for (auto k : rkeys)
					{
						npmi += logTotN - std::log((float)vocabFreqs[k]);
					}
					npmi /= (logTotN - std::log((float)node->val)) * (rkeys.size() - 1);
					if (npmi < minNPMI) return;

					float rbe = branchingEntropy(node, candMinCnt);
					float lbe = branchingEntropy(trieNodesBw[0].findNode(rkeys.rbegin(), rkeys.rend()), candMinCnt);
					float nbe = std::sqrt(rbe * lbe) / (float)std::log(node->val);
					if (nbe < minNBE) return;
					candidates.emplace_back(npmi * nbe, rkeys);
					candidates.back().cf = node->val;
				}, rkeys);
			}

			std::sort(candidates.begin(), candidates.end(), [](const label::Candidate& a, const label::Candidate& b)
			{
				return a.score > b.score;
			});
			if (candidates.size() > maxCandidates) candidates.erase(candidates.begin() + maxCandidates, candidates.end());
			return candidates;
		}
	}
}
