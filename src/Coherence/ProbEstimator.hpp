#pragma once

#include "Common.h"

namespace tomoto
{
	namespace coherence
	{
		namespace detail
		{
			struct WeightedDocId
			{
				size_t docId = 0, weight = 0;

				WeightedDocId() = default;
				WeightedDocId(size_t _docId, size_t _weight)
					: docId{ _docId }, weight{ _weight }
				{
				}

				bool operator==(const WeightedDocId& o) const
				{
					return docId == o.docId;
				}

				bool operator!=(const WeightedDocId& o) const
				{
					return operator==(o);
				}

				bool operator<(const WeightedDocId& o) const
				{
					return docId < o.docId;
				}
			};


			struct CountIter
			{
				size_t count = 0;
				CountIter& operator++() { return *this; }
				
				struct Wrapper
				{
					CountIter& ci;

					Wrapper(CountIter& _ci) : ci{ _ci }
					{
					}

					void operator=(const size_t)
					{
						++ci.count;
					}

					void operator=(const WeightedDocId& t)
					{
						ci.count += t.weight;
					}
				};

				Wrapper operator*() { return { *this }; }

				using value_type = Wrapper;
				using reference = Wrapper;

				using pointer = void;
				using iterator_category = std::forward_iterator_tag;
				using difference_type = std::ptrdiff_t;

			};

			template<ProbEstimation _pe, typename DocIdType>
			class ProbEstimator;

			template<typename DocIdType>
			class ProbEstimator<ProbEstimation::document, DocIdType> : public IProbEstimator
			{
			protected:
				std::unordered_map<Vid, size_t> singleCnt;
				std::unordered_map<VidPair, size_t> jointCnt;
				std::unordered_map<Vid, std::vector<DocIdType>> singleII;
				std::unordered_map<VidPair, std::vector<DocIdType>> jointII;
				size_t totDocs = 0;
			public:
				ProbEstimator() = default;

				ProbEstimator(size_t windowSize)
				{
				}

				template<typename _TargetIter>
				void insertTargets(_TargetIter targetFirst, _TargetIter targetLast)
				{
					for (; targetFirst != targetLast; ++targetFirst)
					{
						singleCnt.emplace(*targetFirst, 0);
					}
				}

				template<typename _TargetIter>
				void insertDoc(_TargetIter wordFirst, _TargetIter wordLast)
				{
					using iter = std::pair<const Vid, size_t>*;
					std::unordered_set<iter> uniqs, negs;
					for (; wordFirst != wordLast; ++wordFirst)
					{
						auto it = singleCnt.find(*wordFirst);
						if (it != singleCnt.end())
						{
							uniqs.emplace(&*it);
						}
					}

					for (auto& u : uniqs)
					{
						u->second++;
						singleII[u->first].emplace_back(totDocs);
						for (auto& v : uniqs)
						{
							if (u->first < v->first)
							{
								jointCnt[{ u->first, v->first }]++;
								jointII[{ u->first, v->first }].emplace_back(totDocs);
							}
						}
					}
					totDocs += 1;
				}

				void updateII()
				{
					for (auto& p : singleII)
					{
						std::sort(p.second.begin(), p.second.end());
					}

					for (auto& p : jointII)
					{
						std::sort(p.second.begin(), p.second.end());
					}
				}

				double getProb(Vid word) const
				{
					auto it = singleCnt.find(word);
					if (it == singleCnt.end()) return 0;
					return it->second / (double)totDocs;
				}

				double getProb(Vid word1, Vid word2) const
				{
					Vid minWord = std::min(word1, word2);
					Vid maxWord = std::max(word1, word2);
					auto it = jointCnt.find(VidPair{ minWord, maxWord });
					if (it == jointCnt.end()) return 0;
					return it->second / (double)totDocs;
				}

				double getProb(const std::vector<Vid>& words) const
				{
					if (words.size() == 0) return 0;
					if (words.size() == 1) return getProb(words[0]);
					if (words.size() == 2) return getProb(words[0], words[1]);

					Vid vFirst = std::min(words[0], words[1]);
					Vid vSecond = std::max(words[0], words[1]);

					auto it = jointII.find(VidPair{ vFirst, vSecond });
					if (it == jointII.end()) return 0;
					std::vector<DocIdType> intersection = it->second;
					for (size_t i = (words.size() % 2) ? 1 : 2; i < words.size(); i += 2)
					{
						Vid vFirst = std::min(words[i], words[i + 1]);
						Vid vSecond = std::max(words[i], words[i + 1]);

						auto it = jointII.find(VidPair{ vFirst, vSecond });
						if (it == jointII.end()) return 0;
						std::vector<DocIdType> intersectionT;
						std::set_intersection(it->second.begin(), it->second.end(),
							intersection.begin(), intersection.end(),
							std::back_inserter(intersectionT)
						);
						intersectionT.swap(intersection);
					}
					return intersection.size() / (double)totDocs;
				}

				double getJointNotProb(Vid word1, Vid word2) const
				{
					std::vector<DocIdType> intersection;
					auto it = singleII.find(word2);
					if (it != singleII.end()) intersection = it->second;
					if (intersection.empty()) return getProb(word1);

					it = singleII.find(word1);
					if (it == singleII.end()) return 0;
					return std::set_difference(it->second.begin(), it->second.end(),
						intersection.begin(), intersection.end(),
						detail::CountIter{}
					).count / (double)totDocs;
				}

				double getJointNotProb(Vid word1, const std::vector<Vid>& word2) const
				{
					std::vector<DocIdType> intersection;

					if (word2.size() == 0) return 0;
					if (word2.size() == 1)
					{
						auto it = singleII.find(word2[0]);
						if (it != singleII.end()) intersection = it->second;
					}
					else if (word2.size() == 2)
					{
						Vid vFirst = std::min(word2[0], word2[1]);
						Vid vSecond = std::max(word2[0], word2[1]);

						auto it = jointII.find(VidPair{ vFirst, vSecond });
						if (it != jointII.end()) intersection = it->second;
					}
					else do
					{
						Vid vFirst = std::min(word2[0], word2[1]);
						Vid vSecond = std::max(word2[0], word2[1]);

						auto it = jointII.find(VidPair{ vFirst, vSecond });
						if (it == jointII.end()) break;
						intersection = it->second;
						for (size_t i = (word2.size() % 2) ? 1 : 2; i < word2.size(); i += 2)
						{
							Vid vFirst = std::min(word2[i], word2[i + 1]);
							Vid vSecond = std::max(word2[i], word2[i + 1]);

							auto it = jointII.find(VidPair{ vFirst, vSecond });
							if (it == jointII.end()) break;
							std::vector<DocIdType> intersectionT;
							std::set_intersection(it->second.begin(), it->second.end(),
								intersection.begin(), intersection.end(),
								std::back_inserter(intersectionT)
							);
							intersectionT.swap(intersection);
						}
					} while (0);

					if (intersection.empty()) return getProb(word1);
					auto it = singleII.find(word1);
					if (it == singleII.end()) return 0;
					return std::set_difference(it->second.begin(), it->second.end(),
						intersection.begin(), intersection.end(),
						detail::CountIter{}
					).count / (double)totDocs;
				}
			};

			template<>
			class ProbEstimator<ProbEstimation::sliding_windows, WeightedDocId>
				: public ProbEstimator<ProbEstimation::document, WeightedDocId>
			{
				size_t windowSize = 0;
				size_t nextDocId = 0;

			public:
				ProbEstimator() = default;

				ProbEstimator(size_t _windowSize) : windowSize{ _windowSize }
				{
				}

				template<typename _TargetIter>
				void insertDoc(_TargetIter wordFirst, _TargetIter wordLast)
				{
					using iter = std::pair<const Vid, size_t>*;
					std::vector<std::pair<uint32_t, Vid>> posVids;
					std::unordered_map<Vid, uint32_t> vidCnts;
					size_t len = wordLast - wordFirst;
					for (size_t i = 0; i < len; ++i)
					{
						auto it = this->singleCnt.find(wordFirst[i]);
						if (it != this->singleCnt.end())
						{
							posVids.emplace_back(i, wordFirst[i]);
						}
					}

					if (posVids.empty()) return;

					size_t start = 0, end = 0, cend = std::min(windowSize, len);
					while (end < posVids.size() && posVids[end].first < windowSize)
					{
						vidCnts[posVids[end].second]++;
						end++;
					}

					while (cend <= len && start < posVids.size())
					{
						size_t startMargin = posVids[start].first - (cend - windowSize);
						size_t endMargin = end < posVids.size() ? (posVids[end].first - cend + 1) : -1;
						size_t cntWindows = std::min(std::min(startMargin, endMargin), len + 1 - cend);
						for (auto& u : vidCnts)
						{
							if (!u.second) continue;
							this->singleII[u.first].emplace_back(nextDocId, cntWindows);
							this->singleCnt[u.first] += cntWindows;
							for (auto& v : vidCnts)
							{
								if (!v.second) continue;
								if (u.first < v.first)
								{
									this->jointII[{u.first, v.first}].emplace_back(nextDocId, cntWindows);
									this->jointCnt[{ u.first, v.first }] += cntWindows;
								}
							}
						}

						cend += cntWindows;
						if (startMargin < endMargin)
						{
							vidCnts[posVids[start].second]--;
							start++;
						}
						else if (startMargin > endMargin)
						{
							vidCnts[posVids[end].second]++;
							end++;
						}
						else
						{
							vidCnts[posVids[start].second]--;
							vidCnts[posVids[end].second]++;
							start++;
							end++;
						}
						nextDocId++;
					}

					this->totDocs += std::max(len, windowSize) - windowSize + 1;
				}
			};

		}

		template<ProbEstimation _pe>
		class ProbEstimator;

		template<>
		class ProbEstimator<ProbEstimation::document> : public detail::ProbEstimator<ProbEstimation::document, size_t>
		{
		public:
			using detail::ProbEstimator<ProbEstimation::document, size_t>::ProbEstimator;
		};

		template<>
		class ProbEstimator<ProbEstimation::sliding_windows> : public detail::ProbEstimator<ProbEstimation::sliding_windows, detail::WeightedDocId>
		{
		public:
			using detail::ProbEstimator<ProbEstimation::sliding_windows, detail::WeightedDocId>::ProbEstimator;
		};
	}
}
