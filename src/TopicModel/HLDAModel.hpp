#pragma once
#include "LDAModel.hpp"
#include "HLDA.h"

/*
Implementation of hLDA using Gibbs sampling by bab2min

* Griffiths, T. L., Jordan, M. I., Tenenbaum, J. B., & Blei, D. M. (2004). Hierarchical topic models and the nested Chinese restaurant process. In Advances in neural information processing systems (pp. 17-24).
*/

namespace tomoto
{
	namespace detail
	{
		struct NCRPNode
		{
			int32_t numCustomers = 0, level = 0;
			int32_t parent = 0, sibling = 0, child = 0;

			DEFINE_SERIALIZER(numCustomers, level, parent, sibling, child);

			NCRPNode* getParent() const
			{
				if (!parent) return nullptr;
				return (NCRPNode*)(this + parent);
			}

			NCRPNode* getSibling() const
			{
				if (!sibling) return nullptr;
				return (NCRPNode*)(this + sibling);
			}

			NCRPNode* getChild() const
			{
				if (!child) return nullptr;
				return (NCRPNode*)(this + child);
			}

			void setSibling(NCRPNode* node)
			{
				sibling = node ? (node - this) : 0;
			}

			NCRPNode* addChild(NCRPNode* newChild)
			{
				auto* orgChild = getChild();
				child = newChild - this;
				newChild->parent = this - newChild;
				newChild->setSibling(orgChild);
				return newChild;
			}

			void removeChild(NCRPNode* del)
			{
				NCRPNode* prev = getChild();
				if (prev == del)
				{
					child = del->getSibling() ? del->getSibling() - this : 0;
					return;
				}

				for (NCRPNode* node = prev->getSibling(); node; node = node->getSibling())
				{
					if (node == del)
					{
						prev->setSibling(node->getSibling());
						return;
					}
					prev = node;
				}

				throw std::runtime_error{ "Cannot find the child" };
			}

			operator bool() const
			{
				return numCustomers || level;
			}

			bool isLeaf(int totLevel) const
			{
				return level == totLevel - 1;
			}

			void dropPathOne()
			{
				NCRPNode* node = this;
				size_t _level = this->level;
				for (size_t i = 0; i <= _level; ++i)
				{
					if (!--node->numCustomers)
					{
						node->level = 0;
						node->getParent()->removeChild(node);
					}
					node = node->getParent();
				}
			}

			void addPathOne()
			{
				NCRPNode* node = this;
				for (size_t i = 0; i <= level; ++i)
				{
					++node->numCustomers;
					node = node->getParent();
				}
			}
		};

		struct NodeTrees
		{
			static constexpr size_t blockSize = 8;
			std::vector<NCRPNode> nodes;
			std::vector<uint8_t> levelBlocks;
			Eigen::Matrix<FLOAT, -1, 1> nodeLikelihoods; // 
			Eigen::Matrix<FLOAT, -1, 1> nodeWLikelihoods; //

			DEFINE_SERIALIZER(nodes, levelBlocks);

			template<bool _MakeNewPath = true>
			void calcNodeLikelihood(FLOAT gamma, size_t levelDepth)
			{
				nodeLikelihoods.resize(nodes.size());
				nodeLikelihoods.array() = -INFINITY;
				updateNodeLikelihood<_MakeNewPath>(gamma, levelDepth, &nodes[0]);
			}

			template<bool _MakeNewPath = true>
			void updateNodeLikelihood(FLOAT gamma, size_t levelDepth, NCRPNode* node, FLOAT weight = 0)
			{
				size_t idx = node - nodes.data();
				const FLOAT pNewNode = _MakeNewPath ? log(gamma / (node->numCustomers + gamma)) : -INFINITY;
				nodeLikelihoods[idx] = weight + ((node->level < levelDepth - 1) ? pNewNode : 0);
				for(auto * child = node->getChild(); child; child = child->getSibling())
				{
					updateNodeLikelihood(gamma, levelDepth, child, weight + log(child->numCustomers / (node->numCustomers + gamma)));
				}
			}

			void markEmptyBlocks()
			{
				for (size_t b = 0; b < levelBlocks.size(); ++b)
				{
					if (!levelBlocks[b]) continue;
					bool filled = std::any_of(nodes.begin() + (b + 1) * blockSize, nodes.begin() + (b + 2) * blockSize, [](const NCRPNode& node)
					{
						return !!node;
					});
					if (!filled) levelBlocks[b] = 0;
				}
			}

			NCRPNode* newNode(size_t level)
			{
				for (size_t b = 0; b < levelBlocks.size(); ++b)
				{
					if (levelBlocks[b] != level) continue;
					for (size_t i = 0; i < blockSize; ++i)
					{
						const size_t id = blockSize + i + b * blockSize;
						if (!nodes[id]) return &nodes[id];
					}
				}

				for (size_t b = 0; b < levelBlocks.size(); ++b)
				{
					if (!levelBlocks[b])
					{
						levelBlocks[b] = level;
						return &nodes[blockSize + b * blockSize];
					}
				}
				nodes.insert(nodes.end(), blockSize, NCRPNode{});
				levelBlocks.emplace_back(level);
				return &nodes[nodes.size() - blockSize];
			}

			template<TermWeight _TW>
			void calcWordLikelihood(FLOAT eta, size_t realV, size_t levelDepth, ThreadPool* pool,
				const DocumentHLDA<_TW>& doc, const std::vector<FLOAT>& newTopicWeights,
				const ModelStateLDA<_TW>& ld)
			{
				nodeWLikelihoods.resize(nodes.size());
				nodeWLikelihoods.setZero();
				std::vector<std::future<void>> futures;
				futures.reserve(levelBlocks.size());

				auto calc = [this, eta, realV, &doc, &ld](size_t threadId, size_t b)
				{
					FLOAT cnt = 0;
					VID prevWord = -1;
					const size_t bStart = blockSize + b * blockSize;
					for (size_t w = 0; w < doc.words.size(); ++w)
					{
						if (doc.words[w] >= realV) break;
						if (doc.Zs[w] != levelBlocks[b]) continue;
						if (doc.words[w] != prevWord)
						{
							if (prevWord != (VID)-1)
							{
								if (cnt == 1) nodeWLikelihoods.segment(bStart, blockSize).array()
									+= (ld.numByTopicWord.col(prevWord).segment(bStart, blockSize).array().template cast<FLOAT>() + eta).log();
								else nodeWLikelihoods.segment(bStart, blockSize).array()
									+= Eigen::lgamma_subt(ld.numByTopicWord.col(prevWord).segment(bStart, blockSize).array().template cast<FLOAT>() + eta, cnt);
							}
							cnt = 0;
							prevWord = doc.words[w];
						}
						cnt += doc.getWordWeight(w);
					}
					if (prevWord != (VID)-1)
					{
						if (cnt == 1) nodeWLikelihoods.segment(bStart, blockSize).array()
							+= (ld.numByTopicWord.col(prevWord).segment(bStart, blockSize).array().template cast<FLOAT>() + eta).log();
						else nodeWLikelihoods.segment(bStart, blockSize).array()
							+= Eigen::lgamma_subt(ld.numByTopicWord.col(prevWord).segment(bStart, blockSize).array().template cast<FLOAT>() + eta, cnt);
					}
					nodeWLikelihoods.segment(bStart, blockSize).array()
						-= Eigen::lgamma_subt(ld.numByTopic.segment(bStart, blockSize).array().template cast<FLOAT>() + realV * eta, (FLOAT)doc.numByTopic[levelBlocks[b]]);
				};

				// we elide the likelihood for root node because its weight applied to all path and can be seen as constant.
				for (size_t b = 0; b < levelBlocks.size(); ++b)
				{
					if (!levelBlocks[b]) continue;
					if(pool) futures.emplace_back(pool->enqueue(calc, b));
					else calc(0, b);
				}
				for (auto& f : futures) f.get();
				updateWordLikelihood<_TW>(eta, realV, levelDepth, doc, newTopicWeights, &nodes[0]);
			}

			template<TermWeight _TW>
			void updateWordLikelihood(FLOAT eta, size_t realV, size_t levelDepth,
				const DocumentHLDA<_TW>& doc, const std::vector<FLOAT>& newTopicWeights,
				detail::NCRPNode* node, FLOAT weight = 0)
			{
				size_t idx = node - nodes.data();
				weight += nodeWLikelihoods[idx];
				nodeLikelihoods[idx] += weight;
				for (size_t l = node->level + 1; l < levelDepth; ++l)
				{
					nodeLikelihoods[idx] += newTopicWeights[l - 1];
				}
				for (auto* child = node->getChild(); child; child = child->getSibling())
				{
					updateWordLikelihood<_TW>(eta, realV, levelDepth, doc, newTopicWeights, child, weight);
				}
			}

			template<TermWeight _TW>
			size_t generateLeafNode(size_t idx, size_t levelDepth, 
				ModelStateLDA<_TW>& ld)
			{
				for (size_t l = nodes[idx].level + 1; l < levelDepth; ++l)
				{
					auto* nnode = newNode(l);
					idx = nodes[idx].addChild(nnode) - nodes.data();
					nodes[idx].level = l;
				}

				if (ld.numByTopic.size() < nodes.size())
				{
					size_t oldSize = ld.numByTopic.rows();
					size_t newSize = std::max(nodes.size(), ((oldSize + oldSize / 2 + 7) / 8) * 8);
					ld.numByTopic.conservativeResize(newSize);
					ld.numByTopicWord.conservativeResize(newSize, Eigen::NoChange);
					ld.numByTopic.segment(oldSize, newSize - oldSize).setZero();
					ld.numByTopicWord.block(oldSize, 0, newSize - oldSize, ld.numByTopicWord.cols()).setZero();
				}
				return idx;
			}
		};
	}

	template<TermWeight _TW>
	struct ModelStateHLDA : public ModelStateLDA<_TW>
	{
		std::shared_ptr<detail::NodeTrees> nt;

		void serializerRead(std::istream& istr)
		{
			ModelStateLDA<_TW>::serializerRead(istr);
			nt = std::make_shared<detail::NodeTrees>();
			nt->serializerRead(istr);
		}

		void serializerWrite(std::ostream& ostr) const
		{
			ModelStateLDA<_TW>::serializerWrite(ostr);
			nt->serializerWrite(ostr);
		}
	};

	template<TermWeight _TW,
		typename _Interface = IHLDAModel,
		typename _Derived = void,
		typename _DocType = DocumentHLDA<_TW>,
		typename _ModelState = ModelStateHLDA<_TW>>
	class HLDAModel : public LDAModel<_TW, flags::shared_state, _Interface,
		typename std::conditional<std::is_same<_Derived, void>::value, HLDAModel<_TW>, _Derived>::type,
		_DocType, _ModelState>
	{
		static constexpr const char* TMID = "hLDA";
	protected:
		using DerivedClass = typename std::conditional<std::is_same<_Derived, void>::value, HLDAModel<_TW>, _Derived>::type;
		using BaseClass = LDAModel<_TW, flags::shared_state, _Interface, DerivedClass, _DocType, _ModelState>;
		friend BaseClass;
		friend typename BaseClass::BaseClass;
		using WeightType = typename BaseClass::WeightType;

		FLOAT gamma;

		void optimizeParameters(ThreadPool& pool, _ModelState* localData, RandGen* rgs)
		{
			// for alphas
			BaseClass::optimizeParameters(pool, localData, rgs);
			// to do: gamma

		}

		// Words of all documents should be sorted by ascending order.
		template<bool _MakeNewPath = true>
		void samplePathes(_DocType& doc, ThreadPool* pool, _ModelState& ld, RandGen& rgs) const
		{
			if(_MakeNewPath) ld.nt->nodes[doc.path.back()].dropPathOne();
			ld.nt->template calcNodeLikelihood<_MakeNewPath>(gamma, this->K);

			std::vector<FLOAT> newTopicWeights(this->K - 1);
			std::vector<WeightType> cntByLevel(this->K);
			VID prevWord = -1;
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) break;
				addWordToOnlyLocal<-1>(ld, doc, w, doc.words[w], doc.Zs[w]);

				if (_MakeNewPath)
				{
					if (doc.words[w] != prevWord)
					{
						std::fill(cntByLevel.begin(), cntByLevel.end(), 0);
						prevWord = doc.words[w];
					}
					size_t level = doc.Zs[w];
					if (level)
					{
						newTopicWeights[level - 1] += log(this->eta + cntByLevel[level]);
						cntByLevel[level] += doc.getWordWeight(w);
					}
				}
			}

			if (_MakeNewPath)
			{
				for (size_t l = 1; l < this->K; ++l)
				{
					newTopicWeights[l - 1] -= math::lgammaT(doc.numByTopic[l] + this->realV * this->eta) - math::lgammaT(this->realV * this->eta);
				}
			}

			ld.nt->template calcWordLikelihood<_TW>(this->eta, this->realV, this->K, pool, doc, newTopicWeights, ld);

			ld.nt->nodeLikelihoods = (ld.nt->nodeLikelihoods.array() - ld.nt->nodeLikelihoods.maxCoeff()).exp();
			sample::prefixSum(ld.nt->nodeLikelihoods.data(), ld.nt->nodeLikelihoods.size());
			size_t newPath = sample::sampleFromDiscreteAcc(ld.nt->nodeLikelihoods.data(),
				ld.nt->nodeLikelihoods.data() + ld.nt->nodeLikelihoods.size(), rgs);

			if(_MakeNewPath) newPath = ld.nt->template generateLeafNode<_TW>(newPath, this->K, ld);
			doc.path.back() = newPath;
			for (size_t l = this->K - 2; l > 0; --l)
			{
				doc.path[l] = doc.path[l + 1] + ld.nt->nodes[doc.path[l + 1]].parent;
			}

			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) break;
				addWordToOnlyLocal<1>(ld, doc, w, doc.words[w], doc.Zs[w]);
			}
			if (_MakeNewPath) ld.nt->nodes[doc.path.back()].addPathOne();
		}

		template<int INC>
		inline void addWordToOnlyLocal(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID level) const
		{
			assert(vid < this->realV);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			auto weight = doc.getWordWeight(pid);

			updateCnt<DEC>(ld.numByTopic[doc.path[level]], INC * weight);
			updateCnt<DEC>(ld.numByTopicWord(doc.path[level], vid), INC * weight);
		}

		template<int INC>
		inline void addWordTo(_ModelState& ld, _DocType& doc, uint32_t pid, VID vid, TID level) const
		{
			assert(vid < this->realV);
			constexpr bool DEC = INC < 0 && _TW != TermWeight::one;
			auto weight = doc.getWordWeight(pid);

			updateCnt<DEC>(doc.numByTopic[level], INC * weight);
			addWordToOnlyLocal<INC>(ld, doc, pid, vid, level);
		}

		FLOAT* getZLikelihoods(_ModelState& ld, const _DocType& doc, size_t docId, size_t vid) const
		{
			const size_t V = this->realV;
			assert(vid < V);
			auto& zLikelihood = ld.zLikelihood;
			zLikelihood = (doc.numByTopic.array().template cast<FLOAT>() + this->alphas.array());
			for (size_t l = 0; l < this->K; ++l)
			{
				zLikelihood[l] *= (ld.numByTopicWord(doc.path[l], vid) + this->eta)
					/ (ld.numByTopic(doc.path[l]) + V * this->eta);
			}
			sample::prefixSum(zLikelihood.data(), zLikelihood.size());
			return &zLikelihood[0];
		}

		void sampleTopics(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs) const
		{
			for (size_t w = 0; w < doc.words.size(); ++w)
			{
				if (doc.words[w] >= this->realV) continue;
				addWordTo<-1>(ld, doc, w, doc.words[w], doc.Zs[w]);
				auto dist = static_cast<const DerivedClass*>(this)->getZLikelihoods(ld, doc, docId, doc.words[w]);
				doc.Zs[w] = sample::sampleFromDiscreteAcc(dist, dist + this->K, rgs);
				addWordTo<1>(ld, doc, w, doc.words[w], doc.Zs[w]);
			}
		}

		void sampleDocument(_DocType& doc, size_t docId, _ModelState& ld, RandGen& rgs, size_t iterationCnt) const
		{
			sampleTopics(doc, docId, ld, rgs);
		}

		template<typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, RandGen* rgs, _DocIter first, _DocIter last)
		{
			for (auto doc = first; doc != last; ++doc)
			{
				samplePathes<>(*doc, pool, *localData, rgs[0]);
			}
			localData->nt->markEmptyBlocks();
		}

		template<typename _DocIter>
		void sampleGlobalLevel(ThreadPool* pool, _ModelState* localData, RandGen* rgs, _DocIter first, _DocIter last) const
		{
			for (auto doc = first; doc != last; ++doc)
			{
				samplePathes<false>(*doc, pool, *localData, rgs[0]);
			}
		}

		template<typename _DocIter>
		double getLLDocs(_DocIter _first, _DocIter _last) const
		{
			double ll = 0;
			auto lgammaAlpha = math::lgammaT(this->alpha);
			for (; _first != _last; ++_first)
			{
				auto& doc = *_first;
				// doc-path distribution
				for (TID l = 1; l < this->K; ++l)
				{
					ll += log(this->globalState.nt->nodes[doc.path[l]].numCustomers / (this->globalState.nt->nodes[doc.path[l - 1]].numCustomers + gamma));
				}

				// doc-level distribution
				ll -= math::lgammaT(doc.getSumWordWeight() + this->alpha * this->K);
				for (TID l = 0; l < this->K; ++l)
				{
					ll += math::lgammaT(doc.numByTopic[l] + this->alpha) - lgammaAlpha;
				}
			}
			ll += math::lgammaT(this->alpha * this->K) * std::distance(_first, _last);
			return ll;
		}

		double getLLRest(const _ModelState& ld) const
		{
			double ll = 0;
			const size_t V = this->realV;
			const size_t K = ld.nt->nodes.size();
			size_t liveK = 0;
			// topic-word distribution
			auto lgammaEta = math::lgammaT(this->eta);
			for (TID k = 0; k < K; ++k)
			{
				if (!ld.nt->nodes[k]) continue;
				++liveK;
				ll -= math::lgammaT(ld.numByTopic[k] + V * this->eta);
				for (VID v = 0; v < V; ++v)
				{
					if (!ld.numByTopicWord(k, v)) continue;
					ll += math::lgammaT(ld.numByTopicWord(k, v) + this->eta) - lgammaEta;
				}
			}
			ll += math::lgammaT(V*this->eta) * liveK;
			return ll;
		}

		void initGlobalState(bool initDocs)
		{
			const size_t V = this->realV;
			if (initDocs)
			{
				this->globalState.numByTopic = Eigen::Matrix<WeightType, -1, 1>::Zero(this->K);
				this->globalState.numByTopicWord = Eigen::Matrix<WeightType, -1, -1>::Zero(this->K, V);
				this->globalState.nt->nodes.resize(detail::NodeTrees::blockSize);
			}
		}

		void prepareDoc(_DocType& doc, WeightType* topicDocPtr, size_t wordSize) const
		{
			sortAndWriteOrder(doc.words, doc.wOrder);
			doc.numByTopic.init(topicDocPtr, this->K);
			doc.Zs = tvector<TID>(wordSize);
			doc.path.resize(this->K);
			for (size_t l = 0; l < this->K; ++l) doc.path[l] = l;
			
			if (_TW != TermWeight::one) doc.wordWeights.resize(wordSize);
		}

		template<bool _Infer>
		void updateStateWithDoc(typename BaseClass::Generator& g, _ModelState& ld, RandGen& rgs, _DocType& doc, size_t i) const
		{
			if (i == 0)
			{
				ld.nt->template calcNodeLikelihood<!_Infer>(gamma, this->K);
				ld.nt->nodeLikelihoods = (ld.nt->nodeLikelihoods.array() - ld.nt->nodeLikelihoods.maxCoeff()).exp();
				sample::prefixSum(ld.nt->nodeLikelihoods.data(), ld.nt->nodeLikelihoods.size());
				size_t newPath = sample::sampleFromDiscreteAcc(ld.nt->nodeLikelihoods.data(),
					ld.nt->nodeLikelihoods.data() + ld.nt->nodeLikelihoods.size(), rgs);

				if (!_Infer) newPath = ld.nt->generateLeafNode(newPath, this->K, ld);
				doc.path.back() = newPath;
				for (size_t l = this->K - 2; l > 0; --l)
				{
					doc.path[l] = doc.path[l + 1] + ld.nt->nodes[doc.path[l + 1]].parent;
				}

				if (!_Infer) ld.nt->nodes[doc.path.back()].addPathOne();
			}

			auto& z = doc.Zs[i];
			auto w = doc.words[i];
			z = g.theta(rgs);
			addWordTo<1>(ld, doc, i, w, z);
		}

		std::vector<size_t> _getTopicsCount() const
		{
			std::vector<size_t> cnt(this->globalState.nt->nodes.size());
			for (auto& doc : this->docs)
			{
				for (size_t i = 0; i < doc.Zs.size(); ++i)
				{
					if (doc.words[i] < this->realV) ++cnt[doc.path[doc.Zs[i]]];
				}
			}
			return cnt;
		}

	public:
		HLDAModel(size_t _levelDepth = 4, FLOAT _alpha = 0.1, FLOAT _eta = 0.01, FLOAT _gamma = 0.1, const RandGen& _rg = RandGen{ std::random_device{}() })
			: BaseClass(_levelDepth, _alpha, _eta, _rg), gamma(_gamma)
		{
			if (_levelDepth == 0 || _levelDepth >= 0x80000000) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong levelDepth value (levelDepth = %zd)", _levelDepth));
			if (_gamma <= 0) THROW_ERROR_WITH_INFO(std::runtime_error, text::format("wrong gamma value (gamma = %f)", _gamma));
			this->globalState.nt = std::make_shared<detail::NodeTrees>();
		}

		size_t getLiveK() const override
		{
			return std::count_if(this->globalState.nt->nodes.begin(), this->globalState.nt->nodes.end(), [](const detail::NCRPNode& n)
			{
				return !!n;
			});
		}

		size_t getK() const override
		{
			return this->globalState.nt->nodes.size();
		}

		size_t getLevelDepth() const override
		{
			return this->K;
		}

		GETTER(Gamma, FLOAT, gamma);
		DEFINE_SERIALIZER_AFTER_BASE(BaseClass, gamma);

		bool isLiveTopic(TID tid) const override
		{
			return this->globalState.nt->nodes[tid];
		}

		size_t getParentTopicId(TID tid) const override
		{
			if (!isLiveTopic(tid)) return (size_t)-1;
			return this->globalState.nt->nodes[tid].parent ? (tid + this->globalState.nt->nodes[tid].parent) : (size_t)-1;
		}

		size_t getNumDocsOfTopic(TID tid) const override
		{
			if (!isLiveTopic(tid)) return 0;
			return this->globalState.nt->nodes[tid].numCustomers;
		}

		size_t getLevelOfTopic(TID tid) const override
		{
			if (!isLiveTopic(tid)) return (size_t)-1;
			return this->globalState.nt->nodes[tid].level;
		}

		std::vector<size_t> getChildTopicId(TID tid) const override
		{
			std::vector<size_t> ret;
			if (!isLiveTopic(tid)) return ret;
			for (auto* node = this->globalState.nt->nodes[tid].getChild(); node; node = node->getSibling())
			{
				ret.emplace_back(node - this->globalState.nt->nodes.data());
			}
			return ret;
		}
	};
}
