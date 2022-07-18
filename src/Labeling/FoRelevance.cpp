#include <unordered_set>
#include <numeric>

#include "FoRelevance.h"
#include "Phraser.hpp"

using namespace tomoto::label;

template<bool reverse = false>
class DocWordIterator
{
	const tomoto::DocumentBase* doc = nullptr;
	size_t n = 0;
public:
	DocWordIterator(const tomoto::DocumentBase* _doc = nullptr, size_t _n = 0)
		: doc{ _doc }, n{ _n }
	{
	}

	tomoto::Vid operator[](size_t i) const
	{
		return doc->words[doc->wOrder.empty() ? (n + i) : doc->wOrder[n + i]];
	}

	tomoto::Vid operator*() const
	{
		return doc->words[doc->wOrder.empty() ? n : doc->wOrder[n]];
	}

	bool operator==(const DocWordIterator& o) const
	{
		return doc == o.doc && n == o.n;
	}

	bool operator!=(const DocWordIterator& o) const
	{
		return !operator==(o);
	}

	DocWordIterator& operator++()
	{
		if (reverse) --n;
		else ++n;
		return *this;
	}

	DocWordIterator operator+(ptrdiff_t o) const
	{
		return { doc, (size_t)((ptrdiff_t)n + o) };
	}

	DocWordIterator operator-(ptrdiff_t o) const
	{
		return { doc, (size_t)((ptrdiff_t)n - o) };
	}
};

class DocWrapper
{
	const tomoto::DocumentBase* doc;
public:
	DocWrapper(const tomoto::DocumentBase* _doc = nullptr)
		: doc{ _doc }
	{
	}

	size_t size() const
	{
		return doc->words.size();
	}

	tomoto::Vid operator[](size_t idx) const
	{
		return doc->words[doc->wOrder.empty() ? idx : doc->wOrder[idx]];
	}

	DocWordIterator<> begin() const
	{
		return { doc, 0 };
	}

	DocWordIterator<> end() const
	{
		return { doc, doc->words.size() };
	}

	DocWordIterator<true> rbegin() const
	{
		return { doc, doc->words.size() };
	}

	DocWordIterator<true> rend() const
	{
		return { doc, 0 };
	}
};

class DocIterator
{
	const tomoto::ITopicModel* tm;
	size_t idx;
public:
	DocIterator(const tomoto::ITopicModel* _tm = nullptr, size_t _idx = 0)
		: tm{ _tm }, idx{ _idx }
	{
	}

	DocWrapper operator*() const
	{
		return { tm->getDoc(idx) };
	}

	DocIterator& operator++()
	{
		++idx;
		return *this;
	}

	bool operator==(const DocIterator& o) const
	{
		return tm == o.tm && idx == o.idx;
	}

	bool operator!=(const DocIterator& o) const
	{
		return tm != o.tm || idx != o.idx;
	}
};

std::vector<Candidate> PMIExtractor::extract(const tomoto::ITopicModel* tm) const
{
	auto& vocabFreqs = tm->getVocabCf();
	auto& vocabDf = tm->getVocabDf();
	auto candidates = phraser::extractPMINgrams(DocIterator{ tm, 0 }, DocIterator{ tm, tm->getNumDocs() }, 
		vocabFreqs, vocabDf,
		candMinCnt, candMinDf, minLabelLen, maxLabelLen, maxCandidates, 0.f,
		normalized
	);
	if (minLabelLen <= 1)
	{
		for (size_t i = 0; i < vocabDf.size(); ++i)
		{
			if (vocabFreqs[i] < candMinCnt) continue;
			if (vocabDf[i] < candMinDf) continue;
			candidates.emplace_back(0.f, i);
		}
	}
	return candidates;
}

std::vector<Candidate> tomoto::label::PMIBEExtractor::extract(const ITopicModel* tm) const
{
	auto& vocabFreqs = tm->getVocabCf();
	auto& vocabDf = tm->getVocabDf();
	auto candidates = phraser::extractPMIBENgrams(DocIterator{ tm, 0 }, DocIterator{ tm, tm->getNumDocs() },
		vocabFreqs, vocabDf,
		candMinCnt, candMinDf, minLabelLen, maxLabelLen, maxCandidates, 
		0.f, 0.f
	);
	if (minLabelLen <= 1)
	{
		for (size_t i = 0; i < vocabDf.size(); ++i)
		{
			if (vocabFreqs[i] < candMinCnt) continue;
			if (vocabDf[i] < candMinDf) continue;
			candidates.emplace_back(0.f, i);
		}
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
		tomoto::Vid curWord = doc->words[doc->wOrder.empty() ? j : doc->wOrder[j]];
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
					tomoto::OptionalLock<_lock> lock{ mtx[(nnode->val - 1) % (pool ? pool->getNumWorkers() : 1)] };
					auto& c = candidates[nnode->val - 1];
					if (c.name.empty() && !doc->origWordPos.empty())
					{
						size_t start = doc->origWordPos[j + 1 - c.w.size()];
						size_t end = doc->origWordPos[j] + doc->origWordLen[j];
						c.names[doc->rawStr.substr(start, end - start)]++;
					}
					c.docIds.emplace(docId);
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

	if (pool && pool->getNumWorkers() > 1)
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

	Matrix wordTopicDist{ tm->getV(), tm->getK() };
	for (size_t i = 0; i < tm->getK(); ++i)
	{
		auto dist = tm->getWidsByTopic(i);
		wordTopicDist.col(i) = Eigen::Map<Vector>{ dist.data(), (Eigen::Index)dist.size() };
	}

	size_t totDocCnt = 0;
	if (windowSize == (size_t)-1)
	{
		totDocCnt = tm->getNumDocs();
	}
	else
	{
		for (size_t i = 0; i < tm->getNumDocs(); ++i)
		{
			size_t s = tm->getDoc(i)->words.size();
			if (s <= windowSize) totDocCnt += 1;
			else totDocCnt += s - windowSize + 1;
		}
	}

	auto calcScores = [&](CandidateEx& c, size_t windowSize)
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

		size_t docCnt = 0;
		Vector wcPMI = Vector::Zero(this->tm->getV());
		for (auto& docId : c.docIds)
		{
			thread_local Eigen::VectorXi bdf(this->tm->getV());
			bdf.setZero();
			auto doc = this->tm->getDoc(docId);
			if (doc->words.size() <= windowSize)
			{
				for (size_t i = 0; i < doc->words.size(); ++i)
				{
					if (doc->words[i] < this->tm->getV()) bdf[doc->words[i]] = 1;
				}
				docCnt++;
				wcPMI += bdf.template cast<Float>();
			}
			else
			{
				auto wit = c.w.begin();
				std::deque<size_t> wpos;
				for (size_t i = 0; i < windowSize; ++i)
				{
					Vid word = doc->words[doc->wOrder.empty() ? i : doc->wOrder[i]];
					if (word < this->tm->getV()) bdf[word]++;

					if (word == *wit)
					{
						if (++wit == c.w.end())
						{
							wpos.emplace_back(i + 1);
							wit = c.w.begin();
						}
					}
					else if (word == c.w[0]) wit = c.w.begin() + 1;
					else wit = c.w.begin();
				}
				if (!wpos.empty())
				{
					docCnt++;
					wcPMI += Eigen::bool2float(bdf.array()).matrix();
				}

				for (size_t i = windowSize; i < doc->words.size(); ++i)
				{
					Vid oword = doc->words[doc->wOrder.empty() ? (i - windowSize) : doc->wOrder[i - windowSize]];
					Vid word = doc->words[doc->wOrder.empty() ? i : doc->wOrder[i]];
					if (oword < this->tm->getV()) bdf[oword]--;
					if (word < this->tm->getV()) bdf[word]++;
					if (!wpos.empty() && wpos.front() - c.w.size() <= i - windowSize)
					{
						wpos.pop_front();
					}

					if (word == *wit)
					{
						if (++wit == c.w.end())
						{
							wpos.emplace_back(i + 1);
							wit = c.w.begin();
						}
					}
					else if (word == c.w[0]) wit = c.w.begin() + 1;
					else wit = c.w.begin();

					if (!wpos.empty())
					{
						docCnt++;
						wcPMI += Eigen::bool2float(bdf.array()).matrix();
					}
				}
			}
		}
		c.scores = wordTopicDist.transpose() *
			((wcPMI.array() + smoothing) * totDocCnt / docCnt / df.cast<Float>()).log().matrix();
	};

	if (pool && pool->getNumWorkers() > 1)
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
					calcScores(candidates[i], windowSize);
				}
			}, g));
		}
		for (auto& f : futures) f.get();
	}
	else
	{
		for (auto& c : candidates)
		{
			calcScores(c, windowSize);
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
