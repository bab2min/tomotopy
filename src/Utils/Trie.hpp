#pragma once
#include <map>
#include <unordered_map>
#include <deque>
#include <functional>
#include <iterator>
#include "serializer.hpp"

namespace tomoto
{
	template<class _Map>
	class ConstAccess : public _Map
	{
	public:
		auto operator[](typename _Map::key_type key) const -> typename _Map::mapped_type
		{
			auto it = this->find(key);
			if (it == this->end()) return {};
			else return it->second;
		}

		auto operator[](typename _Map::key_type key) -> typename _Map::mapped_type&
		{
			auto it = this->find(key);
			if (it == this->end()) return this->emplace(key, typename _Map::mapped_type{}).first->second;
			else return it->second;
		}

		void serializerWrite(std::ostream& os) const
		{
			serializer::writeMany(os, (const _Map&)*this);
		}

		template<typename _Istr>
		void serializerRead(_Istr& is)
		{
			serializer::readMany(is, (_Map&)*this);
		}
	};

	template<class _Map, class _Node>
	class TrieIterator : public _Map::const_iterator
	{
		using Base = typename _Map::const_iterator;
		using Key = typename _Map::key_type;
		const _Node* base = nullptr;
	public:

		TrieIterator(const Base& it, const _Node* _base)
			: Base(it), base(_base)
		{
		}

		std::pair<const Key, const _Node*> operator*() const
		{
			auto p = Base::operator*();
			return std::make_pair(p.first, base + p.second);
		}
	};

	template<class _Key, class _Value, class _KeyStore = ConstAccess<std::map<_Key, int32_t>>, class _Trie = void>
	struct Trie
	{
		using Node = typename std::conditional<std::is_same<_Trie, void>::value, Trie, _Trie>::type;
		using Key = _Key;
		using KeyStore = _KeyStore;
		using iterator = TrieIterator<_KeyStore, Node>;
		_KeyStore next = {};
		_Value val = {};
		int32_t fail = 0;
		uint32_t depth = 0;

		Trie() {}
		~Trie() {}

		Node* getNext(_Key i) const
		{
			return next[i] ? (Node*)this + next[i] : nullptr;
		}

		Node* getFail() const
		{
			return fail ? (Node*)this + fail : nullptr;
		}

		iterator begin() const
		{
			return { next.begin(), (const Node*)this };
		}

		iterator end() const
		{
			return { next.end(), (const Node*)this };
		}

		template<typename _TyIter, typename _FnAlloc>
		Node* build(_TyIter first, _TyIter last, const _Value& _val, _FnAlloc&& alloc)
		{
			if (first == last)
			{
				if (!val) val = _val;
				return this;
			}

			auto v = *first;
			if (!getNext(v))
			{
				next[v] = alloc() - this;
				getNext(v)->depth = depth + 1;
			}
			return getNext(v)->build(++first, last, _val, alloc);
		}

		template<typename _TyIter>
		Node* findNode(_TyIter begin, _TyIter end)
		{
			if (begin == end) return (Node*)this;
			auto n = getNext(*begin);
			if (n) return n->findNode(++begin, end);
			return nullptr;
		}

		template<typename _Fn>
		void traverse_with_keys(_Fn&& fn, std::vector<_Key>& rkeys) const
		{
			fn((Node*)this, rkeys);

			for (auto& p : next)
			{
				if (p.second)
				{
					rkeys.emplace_back(p.first);
					getNext(p.first)->traverse_with_keys(fn, rkeys);
					rkeys.pop_back();
				}
			}
		}

		template<typename _Fn>
		void traverse_with_keys_post(_Fn&& fn, std::vector<_Key>& rkeys) const
		{
			for (auto& p : next)
			{
				if (p.second)
				{
					rkeys.emplace_back(p.first);
					getNext(p.first)->traverse_with_keys_post(fn, rkeys);
					rkeys.pop_back();
				}
			}
			fn((Node*)this, rkeys);
		}

		template<class _Iterator>
		std::pair<Node*, size_t> findMaximumMatch(_Iterator begin, _Iterator end, size_t idxCnt = 0) const
		{
			if (begin == end) return std::make_pair((Node*)this, idxCnt);
			auto n = getNext(*begin);
			if (n)
			{
				auto v = n->findMaximumMatch(++begin, end, idxCnt + 1);
				if (v.first->val) return v;
			}
			return std::make_pair((Node*)this, idxCnt);
		}

		Node* findFail(_Key i) const
		{
			if (!fail) // if this is Root
			{
				return (Node*)this;
			}
			else
			{
				if (getFail()->getNext(i)) // if 'i' node exists
				{
					return getFail()->getNext(i);
				}
				else // or loop for failure of this
				{
					return getFail()->findFail(i);
				}
			}
		}

		void fillFail()
		{
			std::deque<Node*> dq;
			for (dq.emplace_back((Node*)this); !dq.empty(); dq.pop_front())
			{
				auto p = dq.front();
				for (auto&& kv : p->next)
				{
					auto i = kv.first;
					if (!p->getNext(i)) continue;
					p->getNext(i)->fail = p->findFail(i) - p->getNext(i);
					dq.emplace_back(p->getNext(i));

					if (!p->val)
					{
						for (auto n = p; n->fail; n = n->getFail())
						{
							if (n->val)
							{
								p->val = (_Value)-1;
								break;
							}
						}
					}
				}
			}
		}

		void serializerWrite(std::ostream& os) const
		{
			serializer::writeMany(os, next, val, fail, depth);
		}

		template<typename _Istr>
		void serializerRead(_Istr& is)
		{
			serializer::readMany(is, next, val, fail, depth);
		}
	};

	template<class _Key, class _Value, class _KeyStore = ConstAccess<std::map<_Key, int32_t>>>
	struct TrieEx : public Trie<_Key, _Value, _KeyStore, TrieEx<_Key, _Value, _KeyStore>>
	{
		int32_t parent = 0;

		template<typename _TyIter, typename _FnAlloc>
		TrieEx* build(_TyIter first, _TyIter last, const _Value& _val, _FnAlloc&& alloc)
		{
			if (first == last)
			{
				if (!this->val) this->val = _val;
				return this;
			}

			auto v = *first;
			if (!this->getNext(v))
			{
				this->next[v] = alloc() - this;
				this->getNext(v)->parent = -this->next[v];
			}
			return this->getNext(v)->build(++first, last, _val, alloc);
		}

		template<typename _FnAlloc>
		TrieEx* makeNext(const _Key& k, _FnAlloc&& alloc)
		{
			if (!this->next[k])
			{
				this->next[k] = alloc() - this;
				this->getNext(k)->parent = -this->next[k];
				auto f = this->getFail();
				if (f)
				{
					f = f->makeNext(k, std::forward<_FnAlloc>(alloc));
					this->getNext(k)->fail = f - this->getNext(k);
				}
				else
				{
					this->getNext(k)->fail = this - this->getNext(k);
				}
			}
			return this + this->next[k];
		}

		TrieEx* getParent() const
		{
			if (!parent) return nullptr;
			return (TrieEx*)this + parent;
		}
	};
}