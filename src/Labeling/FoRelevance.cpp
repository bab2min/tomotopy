#include <unordered_set>
#include <numeric>

#include "FoRelevance.h"

using namespace tomoto::label;

namespace std
{
	template <>
	struct hash<pair<tomoto::Vid, tomoto::Vid>>
	{
		size_t operator()(const pair<tomoto::Vid, tomoto::Vid>& k) const
		{
			return hash<tomoto::Vid>{}(k.first) ^ hash<tomoto::Vid>{}(k.second);
		}
	};
}

std::vector<Candidate> PMIExtractor::extract(const tomoto::ITopicModel * tm) const
{
	auto& vocabFreqs = tm->getVocabCf();
	auto& vocabDf = tm->getVocabDf();

	// counting unigrams & bigrams
	std::unordered_map<std::pair<Vid, Vid>, size_t> bigramCnt, bigramDf;

	for (size_t i = 0; i < tm->getNumDocs(); ++i)
	{
		std::unordered_set<std::pair<Vid, Vid>> uniqBigram;
		auto doc = tm->getDoc(i);
		Vid prevWord = doc->words[doc->wOrder.empty() ? 0 : doc->wOrder[0]];
		for (size_t j = 1; j < doc->words.size(); ++j)
		{
			Vid curWord = doc->words[doc->wOrder.empty() ? j : doc->wOrder[j]];
			if (vocabFreqs[curWord] >= candMinCnt && vocabDf[curWord] >= candMinDf)
			{
				if (vocabFreqs[prevWord] >= candMinCnt && vocabDf[prevWord] >= candMinDf)
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

	if (maxLabelLen > 2)
	{
		std::unordered_set<std::pair<Vid, Vid>> validPair;
		for (auto& p : bigramCnt)
		{
			if (p.second >= candMinCnt) validPair.emplace(p.first);
		}

		trieNodes.resize(1);
		auto allocNode = [&]() { return trieNodes.emplace_back(), &trieNodes.back(); };

		for (size_t i = 0; i < tm->getNumDocs(); ++i)
		{
			auto doc = tm->getDoc(i);
			if (trieNodes.capacity() < trieNodes.size() + doc->words.size() * 2)
			{
				trieNodes.reserve(std::max(trieNodes.size() + doc->words.size() * 2, trieNodes.capacity() * 2));
			}

			Vid prevWord = doc->words[doc->wOrder.empty() ? 0 : doc->wOrder[0]];
			size_t labelLen = 0;
			auto node = &trieNodes[0];
			if (vocabFreqs[prevWord] >= candMinCnt)
			{
				node = trieNodes[0].makeNext(prevWord, allocNode);
				node->val++;
				labelLen = 1;
			}

			for (size_t j = 1; j < doc->words.size(); ++j)
			{
				Vid curWord = doc->words[doc->wOrder.empty() ? j : doc->wOrder[j]];

				if (vocabFreqs[curWord] < candMinCnt)
				{
					node = &trieNodes[0];
					labelLen = 0;
				}
				else
				{
					if (labelLen >= maxLabelLen)
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

	// calculating PMIs
	std::vector<Candidate> candidates;
	for (auto& p : bigramCnt)
	{
		auto& bigram = p.first;
		if (p.second < candMinCnt) continue;
		if (bigramDf[bigram] < candMinDf) continue;
		auto pmi = std::log(p.second * (float)tm->getN()
			/ vocabFreqs[bigram.first] / vocabFreqs[bigram.second]);
		if (pmi <= 0) continue;
		candidates.emplace_back(pmi, bigram.first, bigram.second);
	}

	if (maxLabelLen > 2)
	{
		std::vector<Vid> rkeys;
		trieNodes[0].traverse_with_keys([&](const TrieEx<Vid, size_t>* node, const std::vector<Vid>& rkeys)
		{
			if (rkeys.size() <= 2 || node->val < candMinCnt) return;
			float n = tm->getN();
			auto pmi = node->val / n;
			for (auto k : rkeys)
			{
				pmi *= n / vocabFreqs[k];
			}
			pmi = std::log(pmi);
			candidates.emplace_back(pmi, rkeys);
		}, rkeys);
	}

	std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b)
	{
		return a.score > b.score;
	});
	if (candidates.size() > maxCandidates) candidates.erase(candidates.begin() + maxCandidates, candidates.end());

	for (size_t i = 0; i < vocabDf.size(); ++i)
	{
		if (vocabFreqs[i] < candMinCnt) continue;
		if (vocabDf[i] < candMinDf) continue;
		candidates.emplace_back(0.f, i);
	}
	return candidates;
}

template<bool _lock>
const Eigen::ArrayXi& FoRelevance::updateContext(size_t docId, const tomoto::DocumentBase* doc, const tomoto::Trie<tomoto::Vid, size_t>* root)
{
	thread_local Eigen::ArrayXi bdf(tm->getV());
	bdf.setZero();
	auto node = root;
	for (size_t j = 0; j < doc->words.size(); ++j)
	{
		size_t t = doc->wOrder.empty() ? j : doc->wOrder[j];
		tomoto::Vid curWord = doc->words[t];
		if (curWord < tm->getV()) bdf[curWord] = 1;
		auto nnode = node->getNext(curWord);
		while (!nnode)
		{
			node = node->getFail();
			if (node) nnode = node->getNext(curWord);
			else break;
		}

		if (nnode)
		{
			node = nnode;
			do
			{
				// the matched candidate is found
				if (nnode->val && nnode->val != (size_t)-1)
				{
					auto& c = candidates[nnode->val - 1];
					tomoto::OptionalLock<_lock> lock{ mtx[(nnode->val - 1) % (pool ? pool->getNumWorkers() : 1)] };
					if (c.name.empty() && !doc->origWordPos.empty())
					{
						size_t start = doc->origWordPos[j + 1 - c.w.size()];
						size_t end = doc->origWordPos[j] + doc->origWordLen[j];
						c.names[doc->rawStr.substr(start, end - start)]++;
					}
					auto& docIds = c.docIds;
					if (docIds.empty() || docIds.back() != docId) docIds.emplace_back(docId);
				}
			} while (nnode = nnode->getFail());
		}
		else
		{
			node = root;
		}
	}
	return bdf;
}

void FoRelevance::estimateContexts()
{
	std::vector<Trie<Vid, size_t>> candTrie(1);
	candTrie.reserve(std::accumulate(candidates.begin(), candidates.end(), 0, [](size_t s, const CandidateEx& c)
	{
		return s + c.w.size() * 2;
	}));
	auto& root = candTrie.front();

	size_t idx = 1;
	for (auto& c : candidates)
	{
		root.build(c.w.begin(), c.w.end(), idx++, [&]()
		{
			candTrie.emplace_back();
			return &candTrie.back();
		});
	}
	root.fillFail();

	Eigen::ArrayXi df = Eigen::ArrayXi::Zero(tm->getV());

	if (pool)
	{
		const size_t groups = pool->getNumWorkers() * 4;
		std::vector<std::future<Eigen::ArrayXi>> futures;
		futures.reserve(groups);
		for (size_t g = 0; g < groups; ++g)
		{
			futures.emplace_back(pool->enqueue([&, groups](size_t, size_t g)
			{
				Eigen::ArrayXi pdf = Eigen::ArrayXi::Zero(tm->getV());
				for (size_t i = g; i < tm->getNumDocs(); i += groups)
				{
					pdf += updateContext<true>(i, tm->getDoc(i), &root);
				}
				return pdf;
			}, g));
		}
		for (auto& f : futures) df += f.get();
	}
	else
	{
		for (size_t i = 0; i < tm->getNumDocs(); ++i)
		{
			df += updateContext<false>(i, tm->getDoc(i), &root);
		}
	}

	Eigen::Matrix<Float, -1, -1> wordTopicDist{ tm->getV(), tm->getK() };
	for (size_t i = 0; i < tm->getK(); ++i)
	{
		auto dist = tm->getWidsByTopic(i);
		wordTopicDist.col(i) = Eigen::Map<Eigen::Matrix<Float, -1, 1>>{ dist.data(), (Eigen::Index)dist.size() };
	}

	auto calcScores = [&](CandidateEx& c)
	{
		if (c.docIds.size() < candMinDf) return;
		if (c.name.empty() && !c.names.empty())
		{
			size_t m = 0;
			for (auto& p : c.names)
			{
				if (p.second > m)
				{
					c.name = p.first;
					m = p.second;
				}
			}
		}

		Eigen::Matrix<Float, -1, 1> wcPMI = Eigen::Matrix<Float, -1, 1>::Zero(this->tm->getV());
		for (auto& docId : c.docIds)
		{
			thread_local Eigen::VectorXi bdf(this->tm->getV());
			bdf.setZero();
			auto doc = this->tm->getDoc(docId);
			for (size_t i = 0; i < doc->words.size(); ++i)
			{
				if (doc->words[i] < this->tm->getV()) bdf[doc->words[i]] = 1;
			}
			wcPMI += bdf.cast<Float>();
		}
		c.scores = wordTopicDist.transpose() *
			((wcPMI.array() + smoothing) * this->tm->getNumDocs() / c.docIds.size() / df.cast<Float>()).log().matrix();
	};

	if (pool)
	{
		const size_t groups = pool->getNumWorkers() * 4;
		std::vector<std::future<void>> futures;
		futures.reserve(groups);
		for (size_t g = 0; g < groups; ++g)
		{
			futures.emplace_back(pool->enqueue([&, groups](size_t, size_t g)
			{
				for (size_t i = g; i < candidates.size(); i += groups)
				{
					calcScores(candidates[i]);
				}
			}, g));
		}
		for (auto& f : futures) f.get();
	}
	else
	{
		for (auto& c : candidates)
		{
			calcScores(c);
		}
	}

	std::vector<CandidateEx> filtered;
	for (auto& c : candidates)
	{
		if (c.docIds.size() >= candMinDf) filtered.emplace_back(std::move(c));
	}
	filtered.swap(candidates);
}

std::vector<std::pair<std::string, float>> FoRelevance::getLabels(tomoto::Tid tid, size_t topK) const
{
	std::vector<std::pair<std::string, float>> scores;

	for (auto& c : candidates)
	{
		std::string labels = c.name;
		if (labels.empty())
		{
			for (auto w : c.w)
			{
				labels += tm->getVocabDict().toWord(w);
				labels.push_back(' ');
			}
			labels.pop_back();
		}
		float s = c.scores[tid] * (1 + mu / (tm->getK() - 1)) - c.scores.sum() * mu / (tm->getK() - 1);
		scores.emplace_back(labels, s);
	}

	std::sort(scores.begin(), scores.end(), [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b)
	{
		return a.second > b.second;
	});
	if (scores.size() > topK) scores.erase(scores.begin() + topK, scores.end());
	return scores;
}
